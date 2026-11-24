"""TF-IDF + LinearSVC baseline (R2: the transformer must beat it by >= 8 macro-F1)."""
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from shifa import config as C
from shifa.data import clusters, load_split
from shifa.metrics import bootstrap_ci
from shifa.preprocess import preprocess, stamp


def build():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
    ])
    return make_pipeline(feats, LinearSVC())


def main():
    tr, va = load_split("train"), load_split("val")
    pipe = build().fit([preprocess(t) for t in tr.text], tr.topic)
    pred = pipe.predict([preprocess(t) for t in va.text])
    f, lo, hi = bootstrap_ci(va.topic, pred, groups=clusters(va))
    print(f"baseline val macro-F1 {f:.3f}  cluster CI [{lo:.3f}, {hi:.3f}]")
    C.ART.mkdir(exist_ok=True)
    joblib.dump(pipe, C.BASELINE)
    (C.ART / "metrics_baseline.json").write_text(json.dumps({"val_macro_f1": f, "ci": [lo, hi], **stamp()}, indent=2))


if __name__ == "__main__":
    main()
