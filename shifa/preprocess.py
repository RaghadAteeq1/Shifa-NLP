"""One versioned bilingual preprocessing module, used identically at train and serve time (Lab 1).

Order matters: Unicode + digits first, then PII masking, then the per-model profile.
Digits are normalised before masking so Arabic-Indic phone numbers are caught.
"""
import hashlib
import json
import re
import unicodedata

VERSION = "1.1.0"

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
_SPACES = re.compile(r"\s+")

# Order matters: more specific patterns first. Each entry is (name, pattern, replacement).
# Digits may be split by spaces or dashes ("055 123 4567", "+966 55-123-4567"), so masking runs on the
# whole number, not on one whitespace token. Keyword context (\1) is kept so the sentence still reads.
_SEP = r"[\s-]?"
PII_PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[EMAIL]"),
    ("IBAN", re.compile(r"\bSA\d{2}(?:\s?[0-9A-Z]){20}\b", re.I), "[IBAN]"),
    ("MRN", re.compile(r"(?<![A-Za-z0-9])MRN[-\s]?\d{5,9}(?!\d)", re.I), "[MRN]"),
    # a bare record number after a file keyword: "رقم ملفي 482913", "file no 482913"
    ("MRN", re.compile(r"((?:رقم\s+)?(?:ال)?ملف(?:ي)?(?:\s+الطبي)?|(?:medical\s+)?(?:file|record)(?:\s+(?:no\.?|number|#))?)"
                       r"(\s*[:#]?\s*)\d{5,9}(?!\d)", re.I), r"\1\2[MRN]"),
    # mobile: 05XXXXXXXX, 5XXXXXXXX, +966 5X..., 00966 5X...
    ("PHONE", re.compile(r"(?<![\d+])(?:(?:\+|00)966" + _SEP + r"|0)?5\d(?:" + _SEP + r"\d){7}(?!\d)"), "[PHONE]"),
    # landline: 01X XXXXXXX (011 Riyadh, 012 Makkah/Jeddah, 013 Eastern, 014/016/017)
    ("PHONE", re.compile(r"(?<![\d+])(?:(?:\+|00)966" + _SEP + r"|0)1[1-7](?:" + _SEP + r"\d){7}(?!\d)"), "[PHONE]"),
    # national ID / iqama: 10 digits starting with 1 or 2, separators allowed
    ("NID", re.compile(r"(?<!\d)[12](?:" + _SEP + r"\d){9}(?!\d)"), "[NID]"),
    # payment card: 15-19 digits. Last, so two separate numbers next to each other are not merged into one.
    ("CARD", re.compile(r"(?<!\d)\d(?:" + _SEP + r"\d){14,18}(?!\d)"), "[CARD]"),
]

PROFILES = ("bert_ar", "multilingual")


def _base(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(_DIGITS)


def mask_pii(text: str) -> str:
    for _, pat, repl in PII_PATTERNS:
        text = pat.sub(repl, text)
    return text


def find_pii(text: str) -> list[str]:
    """PII types present in the text (used by tests to measure recall)."""
    t = _base(text)
    found = []
    for name, pat, repl in PII_PATTERNS:
        if pat.search(t) and name not in found:
            found.append(name)
        t = pat.sub(repl, t)
    return found


def normalize(text: str, profile: str = "bert_ar") -> str:
    """Per-model Arabic profile.

    bert_ar      : strip diacritics + tatweel, unify alef forms (for Arabic-aware encoders).
    multilingual : Unicode + whitespace only (keeps the text natural for the embedder).
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    t = _base(text)
    if profile == "bert_ar":
        t = _DIACRITICS.sub("", t).replace("ـ", "").translate(_ALEF)
    return _SPACES.sub(" ", t).strip()


def preprocess(text: str, profile: str = "bert_ar") -> str:
    """The only entry point models should use."""
    return normalize(mask_pii(_base(text)), profile)


# ---- skew canaries -------------------------------------------------------------------
CANARY_STRING = "CANARY MRN-000000 0500000000"

CANARIES = [
    ("MRN-482913 جوالي 0551234567 ايميلي a.b@x.com", "bert_ar", "[MRN] جوالي [PHONE] ايميلي [EMAIL]"),
    ("هويتي ١٠٢٣٤٥٦٧٨٩", "bert_ar", "هويتي [NID]"),
    ("جوالي 055 123 4567 ورقم ملفي 482913", "bert_ar", "جوالي [PHONE] ورقم ملفي [MRN]"),
    ("+966 55-123-4567 / 0114567890", "multilingual", "[PHONE] / [PHONE]"),
    ("أرجو تأجيل المَوعِد", "bert_ar", "ارجو تاجيل الموعد"),
    ("ــهلا   بك", "bert_ar", "هلا بك"),
    ("Call 0551234567 now", "multilingual", "Call [PHONE] now"),
    ("أرجو تأجيل", "multilingual", "أرجو تأجيل"),
    (CANARY_STRING, "bert_ar", "CANARY [MRN] [PHONE]"),
]


def run_canaries() -> list[str]:
    """Returns a list of failure messages (empty = green)."""
    bad = []
    for raw, profile, expected in CANARIES:
        got = preprocess(raw, profile)
        if got != expected:
            bad.append(f"[{profile}] {raw!r} -> {got!r}, expected {expected!r}")
        if preprocess(got, profile) != got:
            bad.append(f"[{profile}] not idempotent for {raw!r}")
    return bad


def assert_canaries() -> None:
    bad = run_canaries()
    if bad:
        raise RuntimeError("preprocessing canaries failed:\n" + "\n".join(bad))


def fingerprint() -> str:
    """Hash of version + canary outputs. Stored with every trained artefact."""
    outs = [preprocess(r, p) for r, p, _ in CANARIES]
    return hashlib.sha256(json.dumps([VERSION, outs], ensure_ascii=False).encode()).hexdigest()[:16]


def stamp() -> dict:
    return {"preprocess_version": VERSION, "preprocess_fingerprint": fingerprint()}
