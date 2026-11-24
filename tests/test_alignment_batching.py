from shifa.batching import length_bucket_batches, padding_waste, run_bucketed
from shifa.ner_utils import align_labels, fix_bio, spans_to_entities, words_from_subwords


def test_align_first_subword_only():
    word_ids = [None, 0, 0, 1, 2, 2, None]
    assert align_labels(word_ids, [5, 6, 7]) == [-100, 5, -100, 6, 7, -100, -100]


def test_words_from_subwords_roundtrip():
    word_ids = [None, 0, 0, 1, None]
    tags = words_from_subwords(word_ids, [0, 1, 0, 2, 0], 2, {0: "O", 1: "B-DATE", 2: "I-DATE"})
    assert tags == ["B-DATE", "I-DATE"]


def test_fix_bio_and_entities():
    assert fix_bio(["O", "I-DATE", "I-DATE"]) == ["O", "B-DATE", "I-DATE"]
    ents = spans_to_entities(["يوم", "الأحد", "x"], ["B-DATE", "I-DATE", "O"])
    assert ents == [{"type": "DATE", "text": "يوم الأحد"}]


def test_entities_keep_the_original_spelling():
    from shifa.engine import NerTagger
    ner = NerTagger.__new__(NerTagger)                     # no ONNX model needed: stub the tagger
    ner.tag_words = lambda ws: [["O", "O", "B-DATE", "I-DATE"]]
    assert ner.entities("ابي أأجل يوم الأحد") == [{"type": "DATE", "text": "يوم الأحد"}]


def test_batching_preserves_order_and_cuts_padding():
    lengths = [5, 90, 6, 80, 7, 100, 4, 95]
    batches = length_bucket_batches(lengths, max_items=4)
    assert sorted(i for b in batches for i in b) == list(range(8))
    naive = [list(range(0, 4)), list(range(4, 8))]
    assert padding_waste(lengths, batches) < padding_waste(lengths, naive)
    out = run_bucketed(lambda xs: [x * 2 for x in xs], list(range(8)), lengths, max_items=3)
    assert out == [i * 2 for i in range(8)]


def test_batching_token_budget():
    lengths = [100] * 10
    for b in length_bucket_batches(lengths, max_items=64, max_tokens=300):
        assert len(b) * 100 <= 300
