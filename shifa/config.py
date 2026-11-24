"""Single place for paths, label sets and model names."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
REP = ROOT / "reports"

TOPICS = ["appointment", "billing", "lab_results", "prescription"]
DIALECTS = ["gulf", "msa", "en"]
ENT_TYPES = ["PERSON", "DATE", "DEPT", "DRUG", "TEST", "MONEY"]
NER_TAGS = ["O"] + [f"{p}-{t}" for t in ENT_TYPES for p in ("B", "I")]

# Candidates are compared in scripts/tokenizer_audit.py; the winner goes in DECISIONS.md.
CLASSIFIER_CKPT = os.environ.get("SHIFA_CLASSIFIER", "distilbert-base-multilingual-cased")
NER_CKPT = os.environ.get("SHIFA_NER", CLASSIFIER_CKPT)
EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

MAX_LEN = 128           # re-derived from the p95 token length in tokenizer_audit.py
SEED = 7
SHORT_WORDS = 6         # <= this many words counts as a "short" message

TOPIC_FP32 = ART / "topic-fp32"
TOPIC_ONNX_FP32 = ART / "topic-onnx-fp32"
TOPIC_ONNX_INT8 = ART / "topic-onnx-int8"
NER_FP32 = ART / "ner-fp32"
NER_ONNX_FP32 = ART / "ner-onnx-fp32"
INDEX_DIR = ART / "index"
BASELINE = ART / "baseline.joblib"
DIALECT_MODEL = ART / "dialect.joblib"
