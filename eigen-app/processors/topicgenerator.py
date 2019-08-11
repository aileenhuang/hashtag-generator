"""
Class for generating topics
TODO:
* Handle invalid/corrupt files, redundant file names
* Multilingual support
* Phrase-based LDA implementation?????
* Clean up user interface
"""
import gensim
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
FILES_PATH = "../../test-docs"
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS


class TopicGenerator:
    def __init__(
        self, files, n_topics=6, n_iter=1500, random_state=1, n_top_words=8, workers=2
    ):
        self.files = files
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.random_state = random_state
        self.n_top_words = n_top_words
        self.workers = workers

        self._lda_model = None
        self._gensim_model = None

    @property
    def lda_model(self):
        if self._lda_model is None:
            self.generate_topics()
        return self._lda_model

    @property
    def gensim_model(self):
        if self._gensim_model is None:
            self.generate_gensim_topics()
        return self._gensim_model

    def _get_normalized_tokens(self, fname):
        """
        Cleans, lemmatizes, and normalizes text for a file.
        """
        with open(
            path.join(path.dirname(__file__), FILES_PATH, fname), "r", encoding="utf-8"
        ) as f:
            text = f.read().replace(
                "\n", " "
            )  # Read in and replace newlines with space
            doc = nlp(text)
            tokens = [
                token for token in doc if not token.is_stop and token.is_alpha
            ]  # Strip stop words and remove non-alphabetical words
            tokens = [
                token.lemma_.lower() for token in tokens if token.pos_ != "PROPN"
            ]  # Force words that are not proper nouns to be lowercase
            return {fname: tokens}
        return None

    def _get_normalized_corpus(self, files):
        with ThreadPool(processes=NUM_THREADS) as pool:
            file_to_tokens_list = pool.map(self._get_normalized_tokens, self.files)

        file_to_tokens = self._map_to_dict(file_to_tokens_list)
        return file_to_tokens

    def _get_document_term_matrix(self, file_to_tokens):
        """
        Generates a document-term matrix, documents being the list of file names and terms
        being the normalized set of unique tokens.

        Columns: words
        Rows: file names
        Cells: Frequencies
        """
        vec = CountVectorizer()
        f_list = list(file_to_tokens.keys())
        X = vec.fit_transform(
            [" ".join(tokens) for f, tokens in file_to_tokens.items()]
        )
        df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

        for i in range(len(f_list)):
            df = df.rename({i: f_list[i]})  # Reindex rows to file names

        np_matrix = df.to_numpy()
        return np_matrix

    def _map_to_dict(self, file_to_tokens_list):
        file_to_tokens = {}
        for file_to_token in file_to_tokens_list:
            file_to_tokens.update(file_to_token)

        return file_to_tokens

    def _get_all_unique_tokens(self, file_to_tokens):
        all_tokens = []
        for f, tokens in file_to_tokens.items():
            all_tokens.extend(tokens)
        all_tokens = list(set(all_tokens))  # List of unique tokens

        return all_tokens

    def generate_topics(self):
        file_to_tokens = self._get_normalized_corpus(self.files)
        all_tokens = self._get_all_unique_tokens(file_to_tokens)

        np_matrix = self._get_document_term_matrix(file_to_tokens)

        model = LDA(
            n_topics=self.n_topics, n_iter=self.n_iter, random_state=self.random_state
        )
        model.fit(np_matrix)

        doc_topic = model.doc_topic_  # document-topic distributions
        topic_word = model.topic_word_  # topic-word distributions

        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(all_tokens)[np.argsort(topic_dist)][
                : -self.n_top_words : -1
            ]
            print("Topic {}: {}".format(i, " ".join(topic_words)))

        for i in range(len(self.files)):
            print("{} Top topic: {}".format(self.files[i], doc_topic[i].argmax()))

        self._lda_model = model

    def generate_gensim_topics(self):
        file_to_tokens = self._get_normalized_corpus(self.files)
        processed_docs = [tokens for f, tokens in file_to_tokens.items()]

        dictionary = gensim.corpora.Dictionary(processed_docs)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        model = gensim.models.LdaMulticore(
            bow_corpus,
            num_topics=self.n_topics,
            id2word=dictionary,
            passes=self.n_iter,
            workers=self.workers,
        )

        for idx, topic in model.print_topics(-1):
            print("Topic: {}\nWords: {}\n".format(idx, topic))

        self._gensim_model = model


def main(files):
    tg = TopicGenerator(files, n_topics=2)
    tg.generate_gensim_topics()
    tg.generate_topics()


if __name__ == "__main__":
    main(TEST_FILES)
