# Licensed to the Technische Universität Darmstadt under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The Technische Universität Darmstadt
# licenses this file to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from cassis import Cas
import spacy
from spacy.tokens import Doc

from ariadne.classifier import Classifier
from ariadne.contrib.inception_util import create_span_prediction, TOKEN_TYPE

import unicodedata

greek_model_path = "//home/ec2-user/server/models/ner-2103"

# Script detection function
def detect_script(text):
    greek_count, latin_count = 0, 0
    for char in text:
        if '\u0370' <= char <= '\u03FF' or '\u1F00' <= char <= '\u1FFF':
            greek_count += 1
        elif 'A' <= char <= 'Z' or 'a' <= char <= 'z':
            latin_count += 1
    return 'greek' if greek_count > latin_count else 'latin'

#normalize text
def normalize_text(text):
    return unicodedata.normalize('NFD', text)

class SpacySpanClassifier(Classifier):
    
    greek_model_path = "//home/ec2-user/server/models/atlomy_full_pipeline_annotation_131024"

    def __init__(self, model_name: str = "la_core_web_lg", model_directory: Path = None, script='latin'):
        super().__init__(model_directory=model_directory)
        self.model_name = model_name  # Store model_name as an instance attribute
        self.script = script  # Ensure this attribute is initialized
        self.ner_classifier = SpacyNerClassifier(model_name, model_directory, script)
        self._update_model(script)
        
    def _update_model(self, script):
        if script == 'greek':
            self._model = spacy.load(self.greek_model_path, disable=["parser"])
        # Ensure the span categorizer is in the pipeline
            if "spancat" not in self._model.pipe_names:
                raise ValueError("The loaded model does not contain a span categorizer component")
        else:
            # For Latin, we'll use the NER classifier, so no need to load a model here
            pass

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        cas_tokens = cas.select(TOKEN_TYPE)
        words = [cas.get_covered_text(cas_token) for cas_token in cas_tokens]
        text = normalize_text(" ".join(words))
        
        # Detect the script of the incoming text
        detected_script = detect_script(text)
       
        # Update the model if the script has changed
        if detected_script != self.script:
            self._update_model(detected_script)
            self.script = detected_script
            
        if self.script == 'greek':
            self._model.get_pipe("spancat").cfg["threshold"] = 0.2 # Adjust the span categorizer threshold as needed
            doc = self._model(text)
        
            # Find the spans (named entities)
            self._model.get_pipe("spancat")(doc)

            # Get spans from the span categorizer
            spans = doc.spans["sc"]  # Assuming "sc" is the key for span categorizer results
            
            predictions = []
            for i, span in enumerate(spans):
                    begin = cas_tokens[span.start].begin
                    end = cas_tokens[span.end - 1].end
                    score = spans.attrs["scores"][i] if "scores" in spans.attrs else None  # Use the score if available, otherwise default to 1.0
                    label = span.label_
                    prediction = create_span_prediction(cas, layer,feature, begin, end, label, score)
                    cas.add_annotation(prediction)
                    
        else:
            # For Latin, use the NER classifier
            self.ner_classifier.predict(cas, layer, feature, project_id, document_id, user_id)

    
class SpacyNerClassifier(Classifier):
    
    greek_model_path = "//home/ec2-user/server/models/ner-2103"
    
    def __init__(self, model_name: str = "la_core_web_lg", model_directory: Path = None, script='latin'):
        super().__init__(model_directory=model_directory)
        self.model_name = model_name  # Store model_name as an instance attribute
        self.script = script  # Ensure this attribute is initialized
        self._update_model(script)

    def _update_model(self, script):
        if script == 'greek':
            model_path = "//home/ec2-user/server/models/ner-2103"
        else:  # Default to Latin model
            model_path = self.model_name  # Use the provided model name for the Latin model
        self._model = spacy.load(model_path, disable=["parser"])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        cas_tokens = cas.select(TOKEN_TYPE)
        words = [cas.get_covered_text(cas_token) for cas_token in cas_tokens]
        text = normalize_text(" ".join(words))

        # Detect the script of the incoming text
        detected_script = detect_script(text)
        # Update the model if the script has changed
        if detected_script != self.script:
            self._update_model(detected_script)
        doc = self._model(text)
        
        # Find the named entities
        self._model.get_pipe("ner")(doc)
        #print ("ents: ", doc)

        # For every entity returned by spacy, create an annotation in the CAS
        for entity in doc.ents:
            #print ("start: ", entity.start , "end: ", entity.end, "label: ", entity.label_)
            begin = cas_tokens[entity.start].begin
            end = cas_tokens[entity.end - 1].end
            label = entity.label_
            prediction = create_span_prediction(cas, layer, feature, begin, end, label)
            cas.add_annotation(prediction)

class SpacyPosClassifier(Classifier):
    greek_model_path = "//home/ec2-user/server/models/ner-2103"
    
    def __init__(self, model_name: str = "la_core_web_lg", model_directory: Path = None, script='latin'):
        super().__init__(model_directory=model_directory)
        self.model_name = model_name  # Store model_name as an instance attribute
        self.script = script
        self._update_model(script)

    def _update_model(self, script):
        if script == 'greek':
            model_path = "//home/ec2-user/server/models/ner-2103"
        else:  # Default to Latin model
            model_path = self.model_name  # Use the provided model name for the Latin model
        self._model = spacy.load(model_path, disable=["parser"])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        # Extract the tokens from the CAS and create a spacy doc from it
        cas_tokens = cas.select(TOKEN_TYPE)
        words = [cas.get_covered_text(token) for token in cas_tokens]
        text = normalize_text(" ".join(words))

        # Detect the script of the incoming text
        detected_script = detect_script(text)
        # Update the model if the script has changed
        if detected_script != self.script:
            self.script = detected_script
            self._update_model(detected_script)
        doc = self._model(text)            
            
        # For every token, extract the POS tag and create an annotation in the CAS
        for cas_token, spacy_token in zip(cas_tokens, doc):
            # print (print word and pos)
            #print ("word: ", cas_token.get_covered_text(), "pos: ", spacy_token.pos_)
            pos_tag = spacy_token.pos_
            prediction = create_span_prediction(cas, layer, feature, cas_token.begin, cas_token.end, pos_tag)
            cas.add_annotation(prediction)


class SpacyLemmaClassifier(Classifier):
    
    greek_model_path = "//home/ec2-user/server/models/ner-2103"
    
    def __init__(self, model_name: str = "la_core_web_lg", model_directory: Path = None, script='latin'):
        super().__init__(model_directory=model_directory)
        self.model_name = model_name  # Store model_name as an instance attribute
        self.script = script
        self._update_model(script)
    
    def _update_model(self, script):
        # Define the model paths for Greek and Latin models
        if script == 'greek':
            model_path = "//home/ec2-user/server/models/ner-2103"
        else:  # Default to Latin model
            model_path = self.model_name  # Use the provided model name for the Latin model
        # Load the model with specified components disabled
        self._model = spacy.load(model_path, disable=["parser", "ner"])

    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        cas_tokens = cas.select(TOKEN_TYPE)
        words = [cas.get_covered_text(token) for token in cas_tokens]
        text = normalize_text(" ".join(words))

        # Detect the script of the incoming text and update the model if necessary
        detected_script = detect_script(text)
        if detected_script != self.script:
            self.script = detected_script
            self._update_model(detected_script)
      
        # Process the document through the spaCy pipeline
        doc = self._model(text)

        # Extract lemmas for each token and create an annotation in the CAS
        for cas_token, spacy_token in zip(cas_tokens, doc):
            # debug print (print word and lemma)
            #print ("word: ", cas_token.get_covered_text(), "lemma: ", spacy_token.lemma_)
            lemma = spacy_token.lemma_
            prediction = create_span_prediction(cas, layer, feature, cas_token.begin, cas_token.end, lemma)
            cas.add_annotation(prediction)



