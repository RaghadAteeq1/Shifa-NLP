import numpy as np
import pandas as pd

from shifa import metrics as M


def test_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(0)
    y = rng.choice(["a", "b", "c"], 200)
    p = np.where(rng.random(200) < 0.7, y, rng.choice(["a", "b", "c"], 200))
    assert abs(M.macro_f1(y, p) - f1_score(y, p, average="macro")) < 1e-9


def test_ci_contains_point_and_narrows_with_n():
    rng = np.random.default_rng(1)
    def sample(n):
        y = rng.choice(["a", "b"], n)
        p = np.where(rng.random(n) < 0.8, y, rng.choice(["a", "b"], n))
        return y, p
    f, lo, hi = M.bootstrap_ci(*sample(100))
    assert lo <= f <= hi
    f2, lo2, hi2 = M.bootstrap_ci(*sample(2000))
    assert (hi2 - lo2) < (hi - lo)


def test_paired_diff_zero_for_identical_models():
    y = np.array(list("ababababab" * 10))
    p = y.copy()
    d, lo, hi = M.paired_bootstrap_diff(y, p, p)
    assert d == 0 and lo == 0 and hi == 0


def test_paired_diff_detects_real_gain():
    rng = np.random.default_rng(2)
    y = rng.choice(["a", "b", "c"], 600)
    good = np.where(rng.random(600) < 0.9, y, "a")
    bad = np.where(rng.random(600) < 0.6, y, "a")
    d, lo, hi = M.paired_bootstrap_diff(y, bad, good)
    assert lo > 0


def test_sliced_flags_small_slices():
    df = pd.DataFrame({"y": ["a", "b"] * 40, "p": ["a", "b"] * 40, "s": ["big"] * 70 + ["small"] * 10})
    r = M.sliced_report(df, "y", "p", ["s"], min_n=50, n_boot=50)
    assert r.loc[r.value == "small", "flag"].iloc[0] != ""
    assert r.loc[r.value == "big", "flag"].iloc[0] == ""


def test_cluster_bootstrap_is_wider_for_clustered_errors():
    """Errors that come in blocks (one bad template) must give a wider CI than the iid bootstrap claims."""
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(12), 20)                       # 12 templates x 20 messages, like our val split
    y = rng.choice(["a", "b", "c", "d"], len(groups))
    bad = np.isin(groups, [2, 7, 9])                            # three templates the model gets mostly wrong
    p = np.where(bad & (rng.random(len(y)) < 0.8), "a", y)
    _, lo_iid, hi_iid = M.bootstrap_ci(y, p)
    f, lo_cl, hi_cl = M.bootstrap_ci(y, p, groups=groups)
    assert lo_cl <= f <= hi_cl
    assert (hi_cl - lo_cl) > 1.5 * (hi_iid - lo_iid)


def test_cluster_paired_diff_and_slices():
    y = np.array(list("abcd") * 60)
    groups = np.repeat(np.arange(12), 20)
    d, lo, hi = M.paired_bootstrap_diff(y, y, y, groups=groups)
    assert d == lo == hi == 0
    df = pd.DataFrame({"y": y, "p": y, "g": groups, "s": np.where(groups < 3, "few", "many")})
    r = M.sliced_report(df, "y", "p", ["s"], min_n=10, n_boot=50, group_col="g")
    assert int(r.loc[r.value == "many", "clusters"].iloc[0]) == 9
    assert "clusters" in r.loc[r.value == "few", "flag"].iloc[0]


def test_entity_f1_strict():
    t = [["B-DATE", "I-DATE", "O", "B-DEPT"]]
    assert M.entity_f1(t, t)["entity_f1"] == 1.0
    boundary_miss = [["B-DATE", "O", "O", "B-DEPT"]]  # one boundary miss = whole entity wrong
    r = M.entity_f1(t, boundary_miss)
    assert r["per_type"]["DATE"] == 0.0 and r["per_type"]["DEPT"] == 1.0


def test_retrieval_metrics():
    ranks = [1, 2, None, 11]
    assert M.hit_at_k(ranks, 10) == 0.5
    assert abs(M.mrr_at_k(ranks, 10) - (1 + 0.5) / 4) < 1e-9
