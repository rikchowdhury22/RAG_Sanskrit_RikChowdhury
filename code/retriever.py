"""
retriever.py
------------
TF-IDF based retriever using only numpy and Python stdlib.
No external ML dependencies required — fully CPU-based.

Why TF-IDF for Sanskrit?
- Works on character/token level without language-specific tokenizers
- Handles Devanagari Unicode text natively
- Transparent and explainable — important for mixed-script corpora
- Cosine similarity gives reliable ranking for short-to-medium queries
"""

import json
import math
import os
import pickle
import re
from collections import Counter

import numpy as np


def tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Works for both Devanagari and Latin script without language-specific tools.
    Lowercase Latin characters; preserve Devanagari as-is.
    """
    # Split on whitespace and common punctuation
    tokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z]+', text)
    # Lowercase only Latin tokens
    return [t.lower() if t.isascii() else t for t in tokens]


class TFIDFRetriever:
    """
    A TF-IDF retriever backed by numpy cosine similarity.
    Supports index persistence via pickle for fast reload.
    """

    def __init__(self):
        self.chunks: list[dict] = []
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = None
        self.tfidf_matrix: np.ndarray = None  # shape: (n_chunks, vocab_size)

    def build_index(self, chunks: list[dict]) -> None:
        """
        Build TF-IDF index from a list of chunk dicts.

        Args:
            chunks: List of {"chunk_id": str, "source": str, "text": str}
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        tokenized = [tokenize(t) for t in texts]

        # Build vocabulary
        all_tokens = set(tok for doc_tokens in tokenized for tok in doc_tokens)
        self.vocab = {tok: idx for idx, tok in enumerate(sorted(all_tokens))}
        V = len(self.vocab)
        N = len(texts)

        print(f"[retriever] Building index: {N} chunks, {V} vocab tokens")

        # Compute TF matrix (raw term frequency, normalized)
        tf_matrix = np.zeros((N, V), dtype=np.float32)
        for i, doc_tokens in enumerate(tokenized):
            counts = Counter(doc_tokens)
            total = sum(counts.values()) or 1
            for tok, count in counts.items():
                if tok in self.vocab:
                    tf_matrix[i, self.vocab[tok]] = count / total

        # Compute IDF
        df = np.sum(tf_matrix > 0, axis=0)  # document frequency per term
        self.idf = np.log((N + 1) / (df + 1)) + 1  # smoothed IDF

        # TF-IDF matrix
        self.tfidf_matrix = tf_matrix * self.idf

        # L2 normalize each row for cosine similarity
        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid divide-by-zero for empty chunks
        self.tfidf_matrix = self.tfidf_matrix / norms

        print(f"[retriever] Index built successfully.")

    def _vectorize_query(self, query: str) -> np.ndarray:
        """Convert a query string to a normalized TF-IDF vector."""
        tokens = tokenize(query)
        V = len(self.vocab)
        vec = np.zeros(V, dtype=np.float32)
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        for tok, count in counts.items():
            if tok in self.vocab:
                tf = count / total
                vec[self.vocab[tok]] = tf * self.idf[self.vocab[tok]]
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: User query (Sanskrit Devanagari, transliterated, or English)
            top_k: Number of chunks to return

        Returns:
            List of chunk dicts with added "score" field, sorted by relevance
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self._vectorize_query(query)
        # Cosine similarity (dot product since both are L2-normalized)
        scores = self.tfidf_matrix @ query_vec

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results

    def save_index(self, path: str) -> None:
        """Persist index to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "vocab": self.vocab,
                "idf": self.idf,
                "tfidf_matrix": self.tfidf_matrix
            }, f)
        print(f"[retriever] Index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load persisted index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.vocab = data["vocab"]
        self.idf = data["idf"]
        self.tfidf_matrix = data["tfidf_matrix"]
        print(f"[retriever] Index loaded from {path} ({len(self.chunks)} chunks)")
