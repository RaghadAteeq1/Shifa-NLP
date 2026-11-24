"""Two-stage retrieval (Lab 5): bi-encoder + FAISS -> cross-encoder re-rank. Manifest-pinned index."""
import hashlib
import json

import numpy as np

from . import config as C
from . import preprocess as P


def corpus_hash(docs) -> str:
    return hashlib.sha256(json.dumps([d["text"] for d in docs], ensure_ascii=False).encode()).hexdigest()[:16]


class Searcher:
    def __init__(self, index_dir=C.INDEX_DIR, rerank=True):
        import faiss
        from sentence_transformers import CrossEncoder, SentenceTransformer

        self.dir = index_dir
        self.manifest = json.loads((index_dir / "manifest.json").read_text())
        if self.manifest["preprocess_fingerprint"] != P.fingerprint():
            raise RuntimeError("index was built with a different preprocessing version: rebuild with `make index`")
        self.docs = [json.loads(l) for l in (index_dir / "docs.jsonl").read_text(encoding="utf-8").splitlines()]
        if corpus_hash(self.docs) != self.manifest["corpus_hash"]:
            raise RuntimeError("docs.jsonl does not match the manifest")
        self.index = faiss.read_index(str(index_dir / "faiss.index"))
        self.embedder = SentenceTransformer(self.manifest["embedder"])
        self.reranker = CrossEncoder(self.manifest["reranker"]) if rerank else None
        self._faiss = faiss

    def _embed(self, texts):
        v = self.embedder.encode(texts, convert_to_numpy=True).astype("float32")
        self._faiss.normalize_L2(v)          # BOTH sides normalised (the Knowledge-Check trap)
        return v

    def stage1(self, query, n=50):
        q = self._embed([P.preprocess(query, "multilingual")])
        scores, ids = self.index.search(q, n)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def search(self, query, k=5, n_candidates=50, rerank=True, min_score=None):
        cands = self.stage1(query, n_candidates)
        if rerank and self.reranker is not None and cands:
            q = P.preprocess(query, "multilingual")
            rs = self.reranker.predict([(q, self.docs[i]["text"]) for i, _ in cands])
            cands = sorted(zip([i for i, _ in cands], map(float, rs)), key=lambda x: -x[1])
        out = []
        for i, s in cands[:k]:
            if min_score is not None and s < min_score:
                break                          # honest-empty: nothing close enough
            d = self.docs[i]
            out.append({"id": d["id"], "text": d["text"], "topic": d["topic"], "intent": d["intent"], "dialect": d["dialect"],
                        "resolution": d["resolution"], "score": round(s, 4)})
        return out
