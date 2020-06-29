#!/usr/bin/python3
"""
Class for generating topics
"""
import csv
import gensim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        self._doc_topic_words = None

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

    @property
    def doc_topic_words(self):
        if self._doc_topic_words is None:
            self.generate_output()
        return self._doc_topic_words

    def _map_to_dict(self, file_to_tokens_list):
        file_to_tokens = {}
        for file_to_token in file_to_tokens_list:
            file_to_tokens.update(file_to_token)

        return file_to_tokens

    def _get_normalized_tokens(self, fname):
        """
        Cleans, lemmatizes, and normalizes text for one particular file.
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
            return {fname: tokens}
        return None

    def _get_normalized_corpus(self, files):
        """
        Cleans, lemmatizes, and normalizes text for entire corpus passed in.
        """
        # with ThreadPool(processes=NUM_THREADS) as pool:
        #    file_to_tokens_list = pool.map(self._get_normalized_tokens, self.files)

        """Synchronous loop for testing"""
        file_to_tokens_list = []
        for fname in files:
            file_to_tokens_list.append(self._get_normalized_tokens(fname))

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
        lemma_vec = []
        for f, tokens in file_to_tokens.items():
            lemmas = [token.lemma_ for token in tokens]
            lemma_vec.append(" ".join(lemmas))

        X = vec.fit_transform(
            lemma_vec
        )

        df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

        f_list = list(file_to_tokens.keys())
        for i in range(len(f_list)):
            df = df.rename({i: f_list[i]})  # Reindex rows to file names

        np_matrix = df.to_numpy()
        return np_matrix


    def _get_all_unique_lemmas(self, file_to_tokens):
        all_lemmas = []
        for f, tokens in file_to_tokens.items():
            lemmas = [token.lemma_ for token in tokens]
            all_lemmas.extend(lemmas)
        all_lemmas = list(set(all_lemmas))  # List of unique tokens

        return all_lemmas

    def generate_topics(self):
        file_to_tokens = self._get_normalized_corpus(self.files)

        np_matrix = self._get_document_term_matrix(file_to_tokens)
        model = LDA(
            n_topics=self.n_topics, n_iter=self.n_iter, random_state=self.random_state
        )
        model.fit(np_matrix)

        self._lda_model = model

    def _get_lemma_to_tokens(self, file_to_tokens):
        # Map lemmas to tokens for reverse lookup
        lemma_to_tokens = {}
        for f, tokens in file_to_tokens.items():
            for token in tokens:
                if token.lemma_ not in lemma_to_tokens:
                    lemma_to_tokens[token.lemma_] = [token]
                else:
                    lemma_to_tokens[token.lemma_].append(token)

        return lemma_to_tokens

    def generate_output(self):
        """
        Final output--to be displayed on the app
        Columns: documents, top topic, n top words for topic, 2 corresponding sentences per word?
        """
        file_to_tokens = self._get_normalized_corpus(self.files)
        all_lemmas = self._get_all_unique_lemmas(file_to_tokens)

        doc_topic = self.lda_model.doc_topic_  # document-topic distributions
        topic_word = self.lda_model.topic_word_  # topic-word distributions

        # Map lemmas to tokens
        lemma_to_tokens = self._get_lemma_to_tokens(file_to_tokens)

        # Map topic indices to lemmas
        topic_lemmas_map = {}
        for i, topic_dist in enumerate(topic_word):
            top_lemmas = list(np.array(all_lemmas)[np.argsort(topic_dist)][
                : -self.n_top_words : -1
            ])  # list of top lemmas
            top_lemmas_to_tokens = {}
            for lemma in top_lemmas:
                top_tokens = lemma_to_tokens[lemma]  # Grab all tokens corresponding to that lemma
                top_lemmas_to_tokens[lemma] = top_tokens
            topic_lemmas_map[i] = top_lemmas_to_tokens

        # Final output: a map of doc_names to topic number, lemmas assoc. with that topic
        # and the tokens associated with those words
        doc_topic_words = {}
        for i in range(len(self.files)):
            top_topic = doc_topic[i].argmax()
            current_file = self.files[i]
            doc_topic_words[self.files[i]] = {top_topic: topic_lemmas_map[top_topic]}

        self._doc_topic_words = doc_topic_words

    def generate_gensim_topics(self):
        """
        Uses the gensim implementation of LDA.
        """
        doc_lemmas = []  # A 2D array of all lemmas of tokens for each doc
        file_to_tokens = self._get_normalized_corpus(self.files)
        for f, tokens in file_to_tokens.items():
            lemmas = [token.lemma_ for token in tokens]
            doc_lemmas.append(lemmas)

        dictionary = gensim.corpora.Dictionary(doc_lemmas)
        bow_corpus = [dictionary.doc2bow(lemmas) for lemmas in doc_lemmas]
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

    def plot_log_likelihoods(self):
        plt.plot(self.lda_model.loglikelihoods_[5:])
        plt.show()


def main(files):
    """Sample code as a demonstration"""
    tg = TopicGenerator(files, n_topics=6)
    tg.generate_output()
    tg.plot_log_likelihoods()
    # tg.generate_gensim_topics()  # Calls gensim model. Uncomment this to compare output!


if __name__ == "__main__":
    main(TEST_FILES)
