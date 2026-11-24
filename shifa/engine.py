"""Inference wrappers used by the service, the benchmarks and the evaluation (ONNX Runtime)."""
import json
import os
from pathlib import Path

import numpy as np

from . import config as C
from .batching import run_bucketed
from .ner_utils import fix_bio, spans_to_entities, words_from_subwords
from .preprocess import preprocess


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class OnnxEncoder:
    """One ONNX model + its tokenizer. Pads dynamically to the longest item in the batch."""

    def __init__(self, model_dir: Path, max_len=C.MAX_LEN):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("SHIFA_THREADS", "2"))  # keep small: requests run concurrently
        self.sess = ort.InferenceSession(str(self.dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}
        self.tok = AutoTokenizer.from_pretrained(str(self.dir))
        self.labels = json.loads((self.dir / "labels.json").read_text())["labels"]
        self.max_len = max_len

    def _run(self, enc):
        feed = {k: np.asarray(v, dtype=np.int64) for k, v in enc.items() if k in self.names}
        return self.sess.run(None, feed)[0]

    def n_tokens(self, texts):
        return [len(x) for x in self.tok(texts, truncation=True, max_length=self.max_len)["input_ids"]]


class TopicClassifier(OnnxEncoder):
    def predict_proba(self, texts, profile="bert_ar"):
        clean = [preprocess(t, profile) for t in texts]
        enc = self.tok(clean, padding=True, truncation=True, max_length=self.max_len, return_tensors="np")
        return softmax(self._run(enc))

    def predict_proba_bucketed(self, texts, max_items=32, max_tokens=2048):
        clean = [preprocess(t) for t in texts]
        lengths = self.n_tokens(clean)

        def fn(batch):
            enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="np")
            return list(softmax(self._run(enc)))

        return np.array(run_bucketed(fn, clean, lengths, max_items, max_tokens))

    def predict(self, texts):
        return [self.labels[i] for i in self.predict_proba(texts).argmax(-1)]


class NerTagger(OnnxEncoder):
    def tag_words(self, words_list):
        enc = self.tok(words_list, is_split_into_words=True, padding=True, truncation=True,
                       max_length=self.max_len, return_tensors="np")
        logits = self._run(enc)
        id2tag = dict(enumerate(self.labels))
        return [fix_bio(words_from_subwords(enc.word_ids(i), logits[i].argmax(-1), len(w), id2tag))
                for i, w in enumerate(words_list)]

    def entities(self, text):
        """Tags the normalised words, but returns the entity as the patient wrote it ("الأحد", not "الاحد")."""
        words = preprocess(text).split()
        if not words:
            return []
        tags = self.tag_words([words])[0]
        shown = preprocess(text, "multilingual").split()       # same masking, no alef/diacritic folding
        return spans_to_entities(shown if len(shown) == len(words) else words, tags)


class BaselineTopic:
    """TF-IDF + LinearSVC fallback so the service still runs before the transformer is trained."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.labels = list(pipe.classes_)

    def predict_proba(self, texts, profile="bert_ar"):
        return softmax(self.pipe.decision_function([preprocess(t, profile) for t in texts]) * 3)

    def predict_proba_bucketed(self, texts, **_):
        return self.predict_proba(texts)

    def predict(self, texts):
        return list(self.pipe.predict([preprocess(t) for t in texts]))


def load_topic(prefer=("int8", "fp32")):
    """INT8 if exported, else fp32 ONNX, else the baseline. Returns (model, name)."""
    for kind in prefer:
        d = C.TOPIC_ONNX_INT8 if kind == "int8" else C.TOPIC_ONNX_FP32
        if (d / "model.onnx").exists():
            return TopicClassifier(d), f"onnx-{kind}"
    if C.BASELINE.exists():
        import joblib
        return BaselineTopic(joblib.load(C.BASELINE)), "tfidf-baseline"
    raise FileNotFoundError("No topic model found. Run `make baseline` (quick) or `make train export`.")


def load_ner():
    return NerTagger(C.NER_ONNX_FP32) if (C.NER_ONNX_FP32 / "model.onnx").exists() else None
