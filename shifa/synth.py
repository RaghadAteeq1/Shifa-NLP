"""Deterministic synthetic data generator. Splits by PATIENT and by TEMPLATE (see templates.py)."""
import random

from . import config as C
from .preprocess import preprocess
from .templates import GREETINGS, PII_SUFFIX, SLOT_ENTITY, SLOT_VALUES, TEMPLATES

SPLIT_SHARE = {"train": 0.70, "val": 0.15, "test": 0.15}
TEMPLATE_IDS = {"train": list(range(0, 8)), "val": [8, 9, 10], "test": [11, 12, 13]}
STYLE_WEIGHTS = {"gulf": 0.40, "msa": 0.30, "en": 0.30}


def _pii_values(rng):
    return {
        "mrn": f"MRN-{rng.randint(100000, 999999)}",
        "phone": f"05{rng.randint(10000000, 99999999)}",
        "nid": f"{rng.choice('12')}{rng.randint(100000000, 999999999)}",
        "email": f"user{rng.randint(10, 999)}@example.com",
    }


def make_message(rng, split, style, topic):
    idx = rng.choice(TEMPLATE_IDS[split])
    intent, tpl = TEMPLATES[topic][style][idx]
    toks, tags = [], []

    greet = rng.choice(GREETINGS[style])
    for w in greet.split():
        toks.append(w), tags.append("O")

    for piece in tpl.split():
        if piece.startswith("{") and piece.endswith("}"):
            slot = piece[1:-1]
            value = rng.choice(SLOT_VALUES[style][slot]).split()
            ent = SLOT_ENTITY[slot]
            for j, w in enumerate(value):
                toks.append(w)
                tags.append(("B-" if j == 0 else "I-") + ent)
        else:
            assert "{" not in piece, f"slot must be a whole token: {piece!r}"
            toks.append(piece), tags.append("O")

    if rng.random() < 0.30:
        suffix = rng.choice(PII_SUFFIX[style]).format(**_pii_values(rng))
        for w in suffix.split():
            toks.append(w), tags.append("O")

    raw = " ".join(toks)
    words = preprocess(raw, "bert_ar").split()
    assert len(words) == len(tags), (raw, words, tags)  # masking must not change token count
    return {
        "text": raw,
        "words": words,
        "tags": tags,
        "topic": topic,
        "intent": intent,
        "dialect": style,
        "lang": "en" if style == "en" else "ar",
        "template": idx,
        "n_words": len(words),
    }


def build(seed=C.SEED, n_patients=500, n_messages=1400):
    rng = random.Random(seed)
    patients = [f"P{i:04d}" for i in range(n_patients)]
    rng.shuffle(patients)
    cut1 = int(n_patients * SPLIT_SHARE["train"])
    cut2 = cut1 + int(n_patients * SPLIT_SHARE["val"])
    pools = {"train": patients[:cut1], "val": patients[cut1:cut2], "test": patients[cut2:]}

    out = {"train": [], "val": [], "test": []}
    styles, weights = list(STYLE_WEIGHTS), list(STYLE_WEIGHTS.values())
    for split, share in SPLIT_SHARE.items():
        n = round(n_messages * share)
        # exact quotas per dialect (not random draws) so every slice has enough items for a CI
        quota = []
        for st, w in zip(styles, weights):
            quota += [st] * round(n * w)
        quota = (quota + [styles[0]] * n)[:n]
        rng.shuffle(quota)
        for i in range(n):
            style = quota[i]
            topic = C.TOPICS[i % len(C.TOPICS)]  # balanced classes
            rec = make_message(rng, split, style, topic)
            rec["patient"] = rng.choice(pools[split])
            rec["id"] = f"{split}-{i:04d}"
            rec["split"] = split
            out[split].append(rec)
        rng.shuffle(out[split])
    return out
