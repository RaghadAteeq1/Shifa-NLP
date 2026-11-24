"""Lightweight dialect/language ID (Lab 4 idea, no heavy dependency at serve time).

Char n-gram TF-IDF + logistic regression over gulf / msa / en. It is a front end for slices
and reporting. On real traffic, replace it with the CAMeL Tools dialect-ID used in Lab 4.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from . import config as C
from .preprocess import preprocess


def fit(texts, labels):
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=5.0),
    )
    pipe.fit([preprocess(t) for t in texts], labels)
    return pipe


def save(pipe):
    C.ART.mkdir(exist_ok=True)
    joblib.dump(pipe, C.DIALECT_MODEL)


def load():
    return joblib.load(C.DIALECT_MODEL) if C.DIALECT_MODEL.exists() else None


def predict(pipe, texts):
    return list(pipe.predict([preprocess(t) for t in texts]))
