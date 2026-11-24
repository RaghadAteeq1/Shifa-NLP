"""Fine-tune NER (Lab 3B structure): first sub-token carries the tag, the others are -100."""
import json
import os

import numpy as np
from datasets import Dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

from shifa import config as C
from shifa.data import load_split
from shifa.metrics import entity_f1
from shifa.ner_utils import align_labels, fix_bio, words_from_subwords
from shifa.preprocess import assert_canaries, stamp

T2I = {t: i for i, t in enumerate(C.NER_TAGS)}
I2T = {i: t for t, i in T2I.items()}


def main(seed=C.SEED):
    assert_canaries()
    set_seed(seed)
    tr, va = load_split("train"), load_split("val")
    tok = AutoTokenizer.from_pretrained(C.NER_CKPT)

    def encode(batch):
        enc = tok(batch["words"], is_split_into_words=True, truncation=True, max_length=C.MAX_LEN)
        enc["labels"] = [align_labels(enc.word_ids(i), [T2I[t] for t in tags]) for i, tags in enumerate(batch["tags"])]
        return enc

    def prep(df):
        return Dataset.from_dict({"words": list(df.words), "tags": list(df.tags)}).map(
            encode, batched=True, remove_columns=["words", "tags"])

    dtr, dva = prep(tr), prep(va)
    model = AutoModelForTokenClassification.from_pretrained(
        C.NER_CKPT, num_labels=len(C.NER_TAGS), id2label=I2T, label2id=T2I)

    def decode(df, logits):
        enc = tok(list(df.words), is_split_into_words=True, truncation=True, max_length=C.MAX_LEN)
        preds = []
        for i, words in enumerate(df.words):
            preds.append(fix_bio(words_from_subwords(enc.word_ids(i), logits[i].argmax(-1), len(words), I2T)))
        return preds

    args = TrainingArguments(
        output_dir=str(C.ART / "runs_ner"), num_train_epochs=8, learning_rate=float(os.environ.get("SHIFA_LR", "5e-5")),
        per_device_train_batch_size=16, eval_strategy="no", save_strategy="no", seed=seed, report_to="none")
    trainer = Trainer(model=model, args=args, train_dataset=dtr, data_collator=DataCollatorForTokenClassification(tok))
    trainer.train()

    # pad each sample's logits to a list so decode() can index them
    out = trainer.predict(dva).predictions
    pred_tags = decode(va, out)
    res = entity_f1([list(t) for t in va.tags], pred_tags)
    print(f"val entity-F1 {res['entity_f1']:.3f}", res["per_type"])
    C.NER_FP32.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(C.NER_FP32))
    tok.save_pretrained(str(C.NER_FP32))
    (C.NER_FP32 / "labels.json").write_text(json.dumps({"labels": C.NER_TAGS}))
    (C.NER_FP32 / "card.json").write_text(json.dumps(
        {"checkpoint": C.NER_CKPT, "seed": seed, "val_entity_f1": res["entity_f1"], "per_type": res["per_type"], **stamp()}, indent=2))


if __name__ == "__main__":
    main()
