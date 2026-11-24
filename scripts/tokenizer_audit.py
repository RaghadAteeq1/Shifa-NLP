"""Lab 1 (tokenisation + token-count cells), scaled to Shifa: which tokenizer deserves the classifier?

Writes reports/tokenizer_audit.csv. Paste the winner + numbers into DECISIONS.md#tokenizer.
Fertility = tokens per whitespace word (lower is better). UNK rate = share of [UNK] tokens.
"""
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from shifa import config as C
from shifa.data import load_split
from shifa.preprocess import preprocess

CANDIDATES = [
    "distilbert-base-multilingual-cased",
    "xlm-roberta-base",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "aubmindlab/bert-base-arabertv2",
]


def main():
    df = load_split("train")
    rows = []
    for name in CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(name)
        except Exception as e:  # offline / gated
            print(f"skip {name}: {e}")
            continue
        for lang, g in df.groupby("lang"):
            texts = [preprocess(t) for t in g.text]
            enc = tok(texts, add_special_tokens=False)["input_ids"]
            n_tok = np.array([len(x) for x in enc])
            n_words = np.array([len(t.split()) for t in texts])
            unk = sum(i == tok.unk_token_id for x in enc for i in x) / max(1, n_tok.sum())
            rows.append({"tokenizer": name, "lang": lang, "fertility": n_tok.sum() / n_words.sum(),
                         "unk_rate": unk, "p95_len": int(np.percentile(n_tok + 2, 95))})
    out = pd.DataFrame(rows)
    C.REP.mkdir(exist_ok=True)
    out.to_csv(C.REP / "tokenizer_audit.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
