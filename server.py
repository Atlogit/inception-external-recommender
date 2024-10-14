from ariadne.contrib.spacy import SpacyNerClassifier, SpacyPosClassifier, SpacyLemmaClassifier
from ariadne.server import Server
  
server = Server()
# load the model from path

server.add_classifier("spacy_ner", SpacyNerClassifier("en_core_web_sm"))
server.add_classifier("spacy_pos", SpacyPosClassifier("en_core_web_sm"))
server.add_classifier("spacy_lemma", SpacyLemmaClassifier("en_core_web_sm"))

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    server.start()

