import pytest

from shifa import config as C
from shifa.preprocess import preprocess
from shifa.synth import build


@pytest.fixture(scope="module")
def data():
    return build()


def test_deterministic():
    assert build(seed=3, n_messages=80) == build(seed=3, n_messages=80)


def test_sizes_and_balance(data):
    assert [len(data[s]) for s in ("train", "val", "test")] == [980, 210, 210]
    for s in data:
        counts = {t: sum(r["topic"] == t for r in data[s]) for t in C.TOPICS}
        assert max(counts.values()) - min(counts.values()) <= 1


def test_no_patient_leakage(data):
    sets = {s: {r["patient"] for r in rows} for s, rows in data.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])


def test_val_test_use_unseen_templates(data):
    seen = {s: {r["template"] for r in rows} for s, rows in data.items()}
    assert seen["train"] == set(range(0, 8))
    assert seen["val"] == {8, 9, 10} and seen["test"] == {11, 12, 13}


def test_every_split_covers_every_intent_and_entity_type():
    from shifa.templates import SLOT_ENTITY, TEMPLATES
    for topic, styles in TEMPLATES.items():
        for style, rows in styles.items():
            assert len(rows) == 14
            for ids in (range(0, 8), range(8, 11), range(11, 14)):
                intents = {rows[i][0] for i in ids}
                assert len(intents) == 3, (topic, style, list(ids), intents)
    ents = {s: {SLOT_ENTITY[p[1:-1]] for t in TEMPLATES.values() for rows in t.values() for i in ids
                for p in rows[i][1].split() if p.startswith("{")}
            for s, ids in {"val": range(8, 11), "test": range(11, 14)}.items()}
    assert {"DEPT", "TEST", "MONEY", "PERSON", "DATE"} <= ents["val"] and {"DEPT", "TEST", "MONEY", "PERSON", "DATE"} <= ents["test"]


def test_tags_aligned_with_preprocessed_words(data):
    for rows in data.values():
        for r in rows:
            assert r["words"] == preprocess(r["text"]).split()
            assert len(r["words"]) == len(r["tags"])
            assert set(r["tags"]) <= set(C.NER_TAGS)


def test_slices_large_enough_for_cis(data):
    for d in C.DIALECTS:
        assert sum(r["dialect"] == d for r in data["test"]) >= 50
