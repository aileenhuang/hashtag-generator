import argparse
import numpy as np
import pandas as pd
import pdb
import simplejson as json
import spacy

from lda import LDA, datasets
from multiprocessing.pool import ThreadPool
from os import path

nlp = spacy.load("en_core_web_sm")
TEST_FILES = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt", "doc6.txt"]
FILES_PATH = "test-docs"
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS


def get_normalized_tokens(fname):
    with open(path.join(path.dirname(__file__), FILES_PATH, fname), "r", encoding="utf-8") as f:
        text = f.read().replace("\n", " ")  # Read in and replace newlines with space
        doc = nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]  # Strip stop words, lemmatize, and remove non-alphabetical words
        # TODO: Handle caps normalization
        return tokens
    return None


if __name__ == "__main__":
    file_to_tokens = {}
    for f in TEST_FILES:  # TODO: Make this a little more robust with redundant file handling
        tokens = get_normalized_tokens(f)
        file_to_tokens[f] = tokens
