from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import en_core_web_lg


class TextMatcher:

    class Tokenizer:
        nlp = en_core_web_lg.load()

        def __call__(self, text) -> list:
            return [
                token.lemma_ for token in self.nlp(text)
                if not token.is_stop and not token.is_punct
            ]

    def __init__(self, train_data: dict, ngram_range=(1, 3), max_features=10000):
        self.lookup = {
            k: '. '.join(itm for itm in v.values())
            for k, v in train_data.items()
        }
        self.tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            tokenizer=self.Tokenizer(),
            max_features=max_features,
        )
        self.knn = NearestNeighbors(
            n_neighbors=1,
            n_jobs=-1,
        ).fit(self.tfidf.fit_transform(self.lookup.values()).todense())
        self.baseline, _ = self._worker('')

    def _worker(self, user_input: str):
        vec = self.tfidf.transform([user_input]).todense()
        return (itm[0][0] for itm in self.knn.kneighbors(vec))

    def __call__(self, user_input: str) -> str:
        dist, idx = self._worker(user_input)
        if dist != self.baseline:
            return list(self.lookup.keys())[int(idx)]
        else:
            return 'No Match'


if __name__ == '__main__':
    from train_data import archetypes
    from test_data import biographies

    print()
    match = TextMatcher(archetypes)
    for name, description in biographies.items():
        print(f"{name}: {match(description)}")
    print()
