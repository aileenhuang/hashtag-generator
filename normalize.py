"""
Class for generating topics
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import simplejson as json
import spacy

from lda import LDA
from multiprocessing.pool import ThreadPool
from os import path
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")

NUM_THREADS = 6
TEST_FILES = ["doc1.txt", "doc2.txt", "doc3.txt", "doc4.txt", "doc5.txt", "doc6.txt"]
FILES_PATH = "test-docs"
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS


def get_normalized_tokens(fname):
    """
    Normalizes (lemmatizes) text according to LDA's bag-of-words assumption
    """
    with open(
        path.join(path.dirname(__file__), FILES_PATH, fname), "r", encoding="utf-8"
    ) as f:
        text = f.read().replace("\n", " ")  # Read in and replace newlines with space
        doc = nlp(text)
        tokens = [
            token for token in doc if not token.is_stop and token.is_alpha
        ]  # Strip stop words and remove non-alphabetical words
        tokens = [
            token.lemma_.lower() for token in tokens if token.pos_ != "PROPN"
        ]  # Force words that are not proper nouns to be lowercase
        return {fname: tokens}
    return None


def get_document_term_matrix(file_to_tokens):
    """
    Generates a document-term matrix, documents being the list of file names and terms
    being the normalized set of unique tokens.

    Columns: words
    Rows: file names
    Cells: Frequencies
    """
    vec = CountVectorizer()
    f_list = list(file_to_tokens.keys())
    X = vec.fit_transform([" ".join(tokens) for f, tokens in file_to_tokens.items()])
    df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

    for i in range(len(f_list)):
        df = df.rename({i: f_list[i]})  # Reindex rows to file names

    np_matrix = df.to_numpy()
    return np_matrix


def generate_topics(np_matrix, all_tokens):
    model = LDA(n_topics=6, n_iter=2000, random_state=1)
    model.fit(np_matrix)
    topic_word = model.topic_word_
    n_top_words = 8

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(all_tokens)[np.argsort(topic_dist)][:-n_top_words:-1]
        print("Topic {}: {}".format(i, " ".join(topic_words)))
    plt.plot(model.loglikelihoods_)
    plt.show()


def _map_to_dict(file_to_tokens_list):
    file_to_tokens = {}
    for file_to_token in file_to_tokens_list:
        file_to_tokens.update(file_to_token)

    return file_to_tokens


def _get_all_unique_tokens(file_to_tokens):
    all_tokens = []
    for f, tokens in file_to_tokens.items():
        all_tokens.extend(tokens)
    all_tokens = list(set(all_tokens))  # List of unique tokens

    return all_tokens


def main(files):
    with ThreadPool(processes=NUM_THREADS) as pool:
        file_to_tokens_list = pool.map(get_normalized_tokens, files)

    file_to_tokens = _map_to_dict(file_to_tokens_list)
    all_tokens = _get_all_unique_tokens(file_to_tokens)

    np_matrix = get_document_term_matrix(file_to_tokens)
    generate_topics(np_matrix, all_tokens)


if __name__ == "__main__":
    main(TEST_FILES)
