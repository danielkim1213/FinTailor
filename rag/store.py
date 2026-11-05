"""Retrieval memory for reviews / market cases. Default backend: TF-IDF + cosine in numpy
(no GPU, instant). Set [rag] backend = "bge-m3" to use sentence-transformers embeddings.
Swap to Qdrant/pgvector when the corpus grows past ~1e5 docs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import get_config
from ..db import connect, upsert
from ..utils import stable_id, utcnow

_cache: dict = {}


class RagStore:
    def __init__(self):
        self.backend = get_config()["rag"]["backend"]

    def add_many(self, docs: list[dict]) -> int:
        if not docs:
            return 0
        df = pd.DataFrame([{"doc_id": d.get("doc_id") or stable_id(d["kind"], d["text"]), "kind": d["kind"],
                            "text": d["text"], "meta": json.dumps(d.get("meta", {}), default=str),
                            "created_at": utcnow()} for d in docs])
        with connect() as con:
            n = upsert(con, "rag_docs", df)
        _cache.clear()
        return n

    def _load(self):
        if "docs" not in _cache:
            with connect(read_only=True) as con:
                _cache["docs"] = con.execute("select doc_id, kind, text, meta, created_at from rag_docs").df()
        return _cache["docs"]

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.backend == "bge-m3":
            if "st" not in _cache:
                from sentence_transformers import SentenceTransformer
                _cache["st"] = SentenceTransformer("BAAI/bge-m3")
            return np.asarray(_cache["st"].encode(texts, normalize_embeddings=True))
        raise RuntimeError("dense backend not configured")

    def search(self, query: str, k: int = 5, kind: str | None = None) -> list[dict]:
        docs = self._load()
        if kind:
            docs = docs[docs["kind"] == kind]
        if docs.empty or not query.strip():
            return []
        texts = docs["text"].tolist()
        if self.backend == "bge-m3":
            key = ("emb", len(texts), kind)
            if key not in _cache:
                _cache[key] = self._embed(texts)
            sims = _cache[key] @ self._embed([query])[0]
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            key = ("tfidf", len(texts), kind)
            if key not in _cache:
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
                _cache[key] = (vec, vec.fit_transform(texts))
            vec, M = _cache[key]
            sims = (M @ vec.transform([query]).T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        return [{"doc_id": docs.iloc[i]["doc_id"], "text": docs.iloc[i]["text"], "score": float(sims[i]),
                 "meta": json.loads(docs.iloc[i]["meta"])} for i in idx if sims[i] > 0]
