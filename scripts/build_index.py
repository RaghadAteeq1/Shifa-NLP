"""Build the versioned FAISS index of resolved cases from the TRAIN split only (Lab 5)."""
import json

import faiss
from sentence_transformers import SentenceTransformer

from shifa import config as C
from shifa import preprocess as P
from shifa.data import load_split
from shifa.search import corpus_hash
from shifa.templates import RESOLUTIONS


def main():
    P.assert_canaries()
    tr = load_split("train")
    docs = []
    for r in tr.itertuples():
        ar, en = RESOLUTIONS[(r.topic, r.intent)]
        docs.append({"id": r.id, "text": P.preprocess(r.text, "multilingual"), "topic": r.topic,
                     "intent": r.intent, "dialect": r.dialect, "resolution": en if r.lang == "en" else ar})
    emb = SentenceTransformer(C.EMBEDDER)
    vecs = emb.encode([d["text"] for d in docs], convert_to_numpy=True, show_progress_bar=True).astype("float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])   # inner product on unit vectors == cosine
    index.add(vecs)

    C.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(C.INDEX_DIR / "faiss.index"))
    (C.INDEX_DIR / "docs.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    manifest = {"embedder": C.EMBEDDER, "reranker": C.RERANKER, "dim": int(vecs.shape[1]), "n_docs": len(docs),
                "metric": "inner_product_on_L2_normalised", "corpus_hash": corpus_hash(docs),
                "built_from": "data/train.jsonl", **P.stamp()}
    (C.INDEX_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("index built:", manifest)


if __name__ == "__main__":
    main()
