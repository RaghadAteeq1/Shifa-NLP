from shifa import behaviour
from shifa.synth import build


def test_generators_produce_pairs():
    # train covers every slot type (val/test templates hold only some of them)
    val = __import__("pandas").DataFrame(build(n_messages=800)["train"])
    pairs = behaviour.invariance_pairs(val)
    kinds = {p[0] for p in pairs}
    assert {"swap_person", "swap_date", "swap_dept", "swap_drug", "swap_test", "swap_amount", "greeting", "thanks", "add_contact"} <= kinds
    assert all(o != n for _, o, n in pairs)


def test_no_vacuous_pairs():
    """Every pair must still differ after preprocessing, or it passes for free and inflates the rate."""
    from shifa.preprocess import preprocess
    df = __import__("pandas").DataFrame([r for rows in build(n_messages=800).values() for r in rows])
    pairs = behaviour.invariance_pairs(df)
    assert pairs and all(preprocess(o) != preprocess(n) for _, o, n in pairs)
    contact = [n for k, _, n in pairs if k == "add_contact"]
    assert all("[" in preprocess(n) for n in contact)          # the contact detail really is masked


def test_suite_flags_a_bad_model():
    import pandas as pd
    val = pd.DataFrame(build(n_messages=800)["val"])
    flaky = lambda texts: ["billing" if len(t) % 2 else "appointment" for t in texts]   # length-dependent = not invariant
    r = behaviour.run_suite(flaky, val)
    assert r["invariance_rate"] < 0.95 and r["mft_rate"] < 0.9
