"""Fine-tune the topic classifier (Lab 3A structure: tokenise -> Dataset -> Trainer -> evaluate)."""
import json
import os
import sys

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from shifa import config as C
from shifa.data import clusters, load_split
from shifa.metrics import bootstrap_ci
from shifa.preprocess import assert_canaries, preprocess, stamp

LABELS = C.TOPICS
L2I = {l: i for i, l in enumerate(LABELS)}


def main(seed=C.SEED):
    assert_canaries()
    set_seed(seed)
    tr, va = load_split("train"), load_split("val")          # the frozen test split is never opened here
    tok = AutoTokenizer.from_pretrained(C.CLASSIFIER_CKPT)

    def prep(df):
        ds = Dataset.from_dict({"text": [preprocess(t, "bert_ar") for t in df.text],
                                "label": [L2I[t] for t in df.topic]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=C.MAX_LEN), batched=True)

    dtr, dva = prep(tr), prep(va)
    model = AutoModelForSequenceClassification.from_pretrained(
        C.CLASSIFIER_CKPT, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I)

    def metrics(p):
        pred = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro"), "accuracy": float((pred == p.label_ids).mean())}

    args = TrainingArguments(
        output_dir=str(C.ART / "runs_topic"), num_train_epochs=5, learning_rate=float(os.environ.get("SHIFA_LR", "2e-5")),
        per_device_train_batch_size=16, per_device_eval_batch_size=64, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=1, seed=seed, report_to="none",
        fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=args, train_dataset=dtr, eval_dataset=dva,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()

    pred = trainer.predict(dva).predictions.argmax(-1)
    f, lo, hi = bootstrap_ci(va.topic, [LABELS[i] for i in pred], groups=clusters(va))
    print(f"val macro-F1 {f:.3f}  cluster CI [{lo:.3f}, {hi:.3f}]")
    C.TOPIC_FP32.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(C.TOPIC_FP32))
    tok.save_pretrained(str(C.TOPIC_FP32))
    (C.TOPIC_FP32 / "labels.json").write_text(json.dumps({"labels": LABELS}))
    (C.TOPIC_FP32 / "card.json").write_text(json.dumps(
        {"checkpoint": C.CLASSIFIER_CKPT, "seed": seed, "val_macro_f1": f, "val_ci": [lo, hi],
         "max_len": C.MAX_LEN, "train_n": len(tr), **stamp()}, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else C.SEED)
