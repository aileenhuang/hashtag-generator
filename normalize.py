import numpy as np
import pandas as pd
import pdb
import simplejson as json
import spacy

from os import path

nlp = spacy.load("en_core_web_sm")
DOCS_PATH = "test-docs"
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS

def load_document(fname):
    with open(path.join(path.dirname(__file__), DOCS_PATH, fname), "r", encoding="utf-8") as file:
        text = file.read().replace("\n", " ")  # Read in and replace newlines with space
        doc = nlp(text)
        print([chunk for chunk in doc.noun_chunks])
        tokens = [token.text for token in doc if not token.is_stop]  # Strip stop
        " ".join(tokens)
        print(tokens)


if __name__ == "__main__":
    load_document("doc1.txt")
