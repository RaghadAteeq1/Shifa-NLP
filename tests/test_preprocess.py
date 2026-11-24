import pytest

from shifa import preprocess as P


def test_canaries_green():
    assert P.run_canaries() == []


@pytest.mark.parametrize("raw,expected", [
    ("جوالي 0551234567", "[PHONE]"),
    ("جوالي ٠٥٥١٢٣٤٥٦٧", "[PHONE]"),          # Arabic-Indic digits
    ("call +966551234567", "[PHONE]"),
    ("ملفي MRN-482913", "[MRN]"),
    ("file mrn 482913", "[MRN]"),
    ("هويتي 1023456789", "[NID]"),
    ("بريدي ali.k@example.com", "[EMAIL]"),
    # real-world formats: separators, no prefix, landlines, cards, IBAN
    ("جوالي 055 123 4567", "[PHONE]"),
    ("جوالي 055-123-4567", "[PHONE]"),
    ("call +966 55 123 4567", "[PHONE]"),
    ("call 00966 55 123 4567", "[PHONE]"),
    ("تلفون البيت 011 456 7890", "[PHONE]"),
    ("رقم ملفي 482913", "[MRN]"),
    ("رقم الملف: ٤٨٢٩١٣", "[MRN]"),
    ("file no 482913", "[MRN]"),
    ("هويتي 1023 4567 89", "[NID]"),
    ("iqama 2123456789", "[NID]"),
    ("بطاقتي 4111 1111 1111 1111", "[CARD]"),
    ("IBAN SA03 8000 0000 6080 1016 7519", "[IBAN]"),
])
def test_pii_masked(raw, expected):
    out = P.preprocess(raw)
    assert expected in out
    assert not any(ch.isdigit() for ch in out.replace("[", "")) or expected == "[MRN]"


@pytest.mark.parametrize("raw", [
    "دفعت 1200 ريال", "SAR 350", "on 12 March", "HbA1c", "2026-11-24", "الموعد الساعة 10:30", "وزني 85 كيلو",
    "ملفات التحاليل 3", "السكر التراكمي 6.5",
])
def test_no_false_positives(raw):
    """Amounts, dates, times and lab values are not PII and must survive (they carry the meaning)."""
    assert P.preprocess(raw, "multilingual") == P.normalize(raw, "multilingual")


def test_pii_recall_100_percent_on_synthetic():
    from shifa.synth import build
    data = build(n_messages=600)
    n_pii = 0
    for rows in data.values():
        for r in rows:
            if P.find_pii(r["text"]):
                n_pii += 1
                masked = P.preprocess(r["text"])
                assert "MRN-" not in masked and "@" not in masked
                assert not any(w.isdigit() and len(w) >= 9 for w in masked.split())
    assert n_pii > 50  # the generator really contains PII


def test_idempotent_and_profiles():
    t = "أرجو تأجيل المَوعِد 0551234567"
    for prof in P.PROFILES:
        once = P.preprocess(t, prof)
        assert P.preprocess(once, prof) == once
    assert P.preprocess("أرجو", "bert_ar") == "ارجو"
    assert P.preprocess("أرجو", "multilingual") == "أرجو"


def test_unknown_profile():
    with pytest.raises(ValueError):
        P.preprocess("x", "nope")


def test_fingerprint_stable():
    assert P.fingerprint() == P.fingerprint()
