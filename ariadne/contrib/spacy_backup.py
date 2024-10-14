from pathlib import Path
from cassis import Cas

import spacy
from spacy.tokens import Doc

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_prediction, TOKEN_TYPE

import unicodedata

def detect_script(text):
    """
    Detects whether the text is primarily in the Greek or Latin script.
    Returns 'greek' if Greek script is predominant, otherwise 'latin'.
    """
    greek_count = 0
    latin_count = 0

    for char in text:
        if '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF':
            greek_count += 1
        elif 'A' <= char <= 'Z' or 'a' <= char <= 'z':
            latin_count += 1

    return 'greek' if greek_count > latin_count else 'latin'

#normalize text
def normalize_text(text):
    return unicodedata.normalize('NFD', text)

class SpacyNerClassifier(Classifier):
    def __init__(self, model_name: str, model_directory: Path = None, script='latin'):
        super().__init__(model_directory=model_directory)
        self.script = script
        if script == 'greek':
            self._model = spacy.load("path_to_greek_model", disable=["parser"])
        else:  # Default to Latin model
            self._model = spacy.load(model_name, disable=["parser"])
        #self._model = spacy.load(model_name, disable=["parser"])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        cas_tokens = cas.select(TOKEN_TYPE)
        # print one token to see internal structure
        #print ("cas_token: ", cas_tokens[0])
        words = [cas.get_covered_text(cas_token) for cas_token in cas_tokens]
        #print ("words: ", words)

        # words parameter is the list of words in the document, alternatively you can pass in the string directly
        # lets unite the list of words into a string

        doc = self._model(normalize_text(" ".join(words)))
        # Find the named entities
        self._model.get_pipe("ner")(doc)
        #print ("ents: ", doc)

        # For every entity returned by spacy, create an annotation in the CAS
        for named_entity in doc.ents:
            #print ("start: ", named_entity.start , "end: ", named_entity.end, "label: ", named_entity.label_)
            begin = cas_tokens[named_entity.start].begin
            end = cas_tokens[named_entity.end - 1].end
            label = named_entity.label_
            prediction = create_prediction(cas, layer, feature, begin, end, label)
            cas.add_annotation(prediction)


class SpacyPosClassifier(Classifier):
    def __init__(self, model_name: str):
        super().__init__()
        self._model = spacy.load(model_name, disable=["parser"])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        words = [cas.get_covered_text(cas_token) for cas_token in cas.select(TOKEN_TYPE)]
        
        doc = self._model(normalize_text(" ".join(words)))
        #doc = Doc(self._model.vocab, words=words)

        # Get the pos tags
        self._model.get_pipe("transformer")(doc)
        self._model.get_pipe("morphologizer")(doc)
        try:
            self._model.get_pipe("attribute_ruler")(doc)
        except:
            print("no attribute ruler found")
            
            
        # For every token, extract the POS tag and create an annotation in the CAS
        for cas_token, spacy_token in zip(cas.select(TOKEN_TYPE), doc):
            # print (print word and pos)
            #print ("word: ", cas_token.get_covered_text(), "pos: ", spacy_token.pos_)
            prediction = create_prediction(cas, layer, feature, cas_token.begin, cas_token.end, spacy_token.pos_)
            cas.add_annotation(prediction)


class SpacyLemmaClassifier(Classifier):
    def __init__(self, model_name: str, model_directory: Path = None):
        super().__init__(model_directory=model_directory)
        self._model = spacy.load(model_name, disable=["parser", "ner"])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        words = [cas.get_covered_text(cas_token) for cas_token in cas.select(TOKEN_TYPE)]

        doc = self._model(normalize_text(" ".join(words)))
        # find the lemmas
        self._model.get_pipe("lemmatizer")(doc)

        # for every token, extract the lemma and create an annotation in the CAS
        for cas_token, spacy_token in zip(cas.select(TOKEN_TYPE), doc):
            # debug print (print word and lemma)
            #print ("word: ", cas_token.get_covered_text(), "lemma: ", spacy_token.lemma_)
            prediction = create_prediction(cas, layer, feature, cas_token.begin, cas_token.end, spacy_token.lemma_)
            cas.add_annotation(prediction)



