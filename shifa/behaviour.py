"""Behavioural tests (Lab 6): invariance pairs + minimum-functionality tests (MFT)."""
import random

from .preprocess import preprocess
from .templates import SLOT_ENTITY, SLOT_VALUES

GREET = {"gulf": "هلا والله", "msa": "السلام عليكم ورحمة الله", "en": "Hello there"}
THANKS = {"gulf": "مشكور", "msa": "شاكر لكم حسن تعاونكم", "en": "thank you"}

# Wordings that appear in NO training template.
MFT = [
    ("ابغى اغير يوم موعدي", "appointment"), ("أحتاج حجز كشف عند الطبيب", "appointment"),
    ("Could you postpone my visit please", "appointment"), ("I will not attend tomorrow so cancel the slot", "appointment"),
    ("المبلغ انخصم مرتين من حسابي", "billing"), ("أريد فاتورة ضريبية عن الزيارة", "billing"),
    ("Please return the money I paid twice", "billing"), ("Send me the receipt of my payment", "billing"),
    ("وين اشوف نتيجة الفحص حقي", "lab_results"), ("أريد تقرير الأشعة والتحاليل", "lab_results"),
    ("Is my blood test finished yet", "lab_results"), ("My lab values look strange to me", "lab_results"),
    ("خلص الدواء عندي وابي وصفة جديدة", "prescription"), ("هل أتناول الحبة قبل النوم أم بعده", "prescription"),
    ("I need more tablets from my doctor", "prescription"), ("This medicine makes me feel sick", "prescription"),
    ("ابي ارجع موعد الاسنان لاسبوع ثاني", "appointment"), ("أطلب إعادة المبلغ المدفوع بالخطأ", "billing"),
    ("متى يجهز تحليل الدم", "lab_results"), ("أرغب في تجديد دواء الضغط", "prescription"),
    ("I want to see a dermatologist next week", "appointment"), ("Why was I billed for a visit I cancelled", "billing"),
    ("Can you send my test report", "lab_results"), ("What is the right dose for my pills", "prescription"),
]


ENT_TO_SLOT = {v: k for k, v in SLOT_ENTITY.items()}
SWAP_KINDS = {"PERSON": "swap_person", "DATE": "swap_date", "DEPT": "swap_dept", "DRUG": "swap_drug",
              "TEST": "swap_test", "MONEY": "swap_amount"}
# Contact details a patient might add. After masking they become [PHONE]/[EMAIL]/[MRN], so the text really changes.
CONTACT = {"gulf": "وهذا جوالي 0551234567", "msa": "ورقم ملفي 482913", "en": "you can reach me at ali@example.com"}


def _swap_entity(words, tags, ent, new_words):
    """Replace the FIRST span of type `ent` with new_words. Returns None if the type is absent."""
    out, i, done = [], 0, False
    while i < len(words):
        if not done and tags[i] == f"B-{ent}":
            j = i + 1
            while j < len(words) and tags[j] == f"I-{ent}":
                j += 1
            out += new_words
            i, done = j, True
        else:
            out.append(words[i])
            i += 1
    return out if done else None


def invariance_pairs(df, seed=0):
    """(kind, original, perturbed). The predicted topic must not change.

    A pair whose two sides are identical AFTER preprocessing tests nothing (it always passes), so it is dropped.
    This is what the old swap_mrn pair did: both record numbers were masked to the same [MRN].
    """
    rng = random.Random(seed)
    pairs = []
    for r in df.itertuples():
        words, tags = list(r.words), list(r.tags)
        base = " ".join(words)
        # 1) swapping any entity value (doctor, date, department, drug, test, amount) must not change the topic
        for ent, kind in SWAP_KINDS.items():
            if f"B-{ent}" not in tags:
                continue
            cur = " ".join(w for w, t in zip(words, tags) if t[2:] == ent)
            others = [v for v in SLOT_VALUES[r.dialect][ENT_TO_SLOT[ent]] if preprocess(v) != preprocess(cur)]
            swapped = _swap_entity(words, tags, ent, rng.choice(others).split())
            if swapped:
                pairs.append((kind, base, " ".join(swapped)))
        # 2) politeness must not change the topic
        pairs.append(("greeting", base, f"{GREET[r.dialect]} {base}"))
        pairs.append(("thanks", base, f"{base} {THANKS[r.dialect]}"))
        # 3) adding contact details must not change the topic (it survives masking as a placeholder)
        pairs.append(("add_contact", base, f"{base} {CONTACT[r.dialect]}"))
    return [p for p in pairs if preprocess(p[1]) != preprocess(p[2])]


def run_suite(predict, df, seed=0):
    """predict: list[str] -> list[str]. Returns rates + the failures (for the error taxonomy)."""
    pairs = invariance_pairs(df, seed)
    a = predict([p[1] for p in pairs])
    b = predict([p[2] for p in pairs])
    inv_fail = [(k, o, n, x, y) for (k, o, n), x, y in zip(pairs, a, b) if x != y]
    by_kind = {}
    for (k, *_), x, y in zip(pairs, a, b):
        t, f = by_kind.get(k, (0, 0))
        by_kind[k] = (t + 1, f + (x != y))
    mft_pred = predict([t for t, _ in MFT])
    mft_fail = [(t, y, p) for (t, y), p in zip(MFT, mft_pred) if p != y]
    return {
        "invariance_rate": 1 - len(inv_fail) / len(pairs), "invariance_n": len(pairs),
        "invariance_by_kind": {k: 1 - f / t for k, (t, f) in by_kind.items()},
        "mft_rate": 1 - len(mft_fail) / len(MFT), "mft_n": len(MFT),
        "invariance_failures": inv_fail, "mft_failures": mft_fail,
    }
