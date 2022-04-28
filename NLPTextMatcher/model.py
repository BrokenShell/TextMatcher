""" TextMatcher: Primary Interface
A machine learning model featuring natural language processing for topic
classification of arbitrary text. TextMatcher creates a callable object for
making predictions. The instance is trained at initialization and the callable
object is reusable for the same training data. TextMatcher uses a combination
of SpaCy, TfidfVectorizer and NearestNeighbors. """
from sys import stderr
from typing import Tuple, List

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print('Downloading language model for the spaCy', file=stderr)
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')


__all__ = ("TextMatcher", "Tokenizer")


class Tokenizer:
    """ Helper Class: Standard SpaCy Tokenizer
    Creates a callable object for tokenizing input data based on the
    SpaCy library. """

    def __call__(self, text: str) -> List[str]:
        """ Callable object for tokenizing text.
        @param text: String of text to be tokenized
        @return: List of tokens as strings derived from SpaCy lemmatization.
        Common stop words are removed and punctuation is ignored. """
        return [
            token.lemma_ for token in nlp(text)
            if not token.is_stop and not token.is_punct
        ]


class TextMatcher:
    """ Primary Interface:
    A machine learning model featuring natural language processing for
    topic classification of arbitrary text.
    Features SpaCy, TfidfVectorizer and NearestNeighbors """

    def __init__(self, train_data: dict, ngram_range=(1, 3), max_features=5000):
        """ Class Initializer
        @param train_data: Dictionary of targets and supporting data. See
        training_data.py for an example.
        @param ngram_range: Tuple representing the range of phrase sizes that
        the model will recognize.
        @param max_features: The maximum number of tokens, this is used to
        fine tune the maximum amount of RAM that the model is allowed to use
        for training. """
        target_lookup = {
            k: " ".join(v.values()) for k, v in train_data.items()
        }
        self.targets = tuple(target_lookup.keys())
        self.tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            tokenizer=Tokenizer(),
            max_features=max_features,
        )
        self.knn = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        vec = self.tfidf.fit_transform(target_lookup.values())
        self.knn.fit(vec.todense())
        self.baseline, _ = self._worker("")

    def _worker(self, user_input: str) -> Tuple[float, int]:
        """ Prediction worker method - internal only
        @param user_input: Arbitrary string of text to be classified.
        @return: Tuple[float, int]: (distance from target, index of target) """
        vec = self.tfidf.transform([user_input])
        dist, idx = self.knn.kneighbors(vec.todense())
        return float(dist), int(idx)

    def __call__(self, user_input: str) -> str:
        """ Callable object for making predictions
        @param user_input: Arbitrary string of text to be classified.
        @return Name of the predicted classification as a string. """
        dist, idx = self._worker(user_input)
        if dist != self.baseline:
            return self.targets[idx]
        else:
            return "No Match"
