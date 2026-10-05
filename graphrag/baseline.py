"""A plain 'naive RAG' baseline: find the sentences most similar to the question (TF-IDF).
Used to show where GraphRAG does better (multi-hop questions)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NaiveRAG:
    def __init__(self, sentences):
        self.sentences = sentences
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = self.vec.fit_transform([s["text"] for s in sentences])

    def search(self, question, k=3):
        q = self.vec.transform([question])
        scores = cosine_similarity(q, self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [self.sentences[i] for i in order if scores[i] > 0]
