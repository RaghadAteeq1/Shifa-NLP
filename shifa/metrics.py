"""Metrics with uncertainty (Lab 6): macro-F1, bootstrap CIs, paired difference, slices, entity-F1, MRR."""
import numpy as np
import pandas as pd


def _codes(y_true, y_pred):
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    idx = {l: i for i, l in enumerate(labels)}
    return (np.array([idx[str(v)] for v in y_true]), np.array([idx[str(v)] for v in y_pred]), len(labels))


def _macro_f1_codes(t, p, k):
    tp = np.bincount(t[t == p], minlength=k).astype(float)
    pred = np.bincount(p, minlength=k).astype(float)
    true = np.bincount(t, minlength=k).astype(float)
    denom = pred + true
    f1 = np.where(denom > 0, 2 * tp / np.where(denom == 0, 1, denom), 0.0)
    present = true > 0
    return float(f1[present].mean())


def macro_f1(y_true, y_pred) -> float:
    t, p, k = _codes(y_true, y_pred)
    return _macro_f1_codes(t, p, k)


def _resampler(N, groups, rng):
    """Index sampler. groups=None: resample items (iid). Otherwise resample whole CLUSTERS with replacement.

    Messages generated from the same template are not independent: they share every word except the slot values.
    Resampling items would pretend we have N independent wordings when we have only a handful, so the
    interval comes out too narrow. Cluster resampling keeps that dependence and gives an honest (wider) CI.
    """
    if groups is None:
        return lambda: rng.integers(0, N, N)
    groups = np.asarray([str(g) for g in groups])
    if len(groups) != N:
        raise ValueError("groups must have one entry per item")
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    G = len(members)
    return lambda: np.concatenate([members[j] for j in rng.integers(0, G, G)])


def n_clusters(groups) -> int:
    return len(set(map(str, groups)))


def bootstrap_ci(y_true, y_pred, n=1000, alpha=0.05, seed=0, groups=None):
    """Returns (point, lo, hi). Resamples test items, or whole clusters when `groups` is given."""
    t, p, k = _codes(y_true, y_pred)
    rng = np.random.default_rng(seed)
    draw = _resampler(len(t), groups, rng)
    stats = np.empty(n)
    for b in range(n):
        ix = draw()
        stats[b] = _macro_f1_codes(t[ix], p[ix], k)
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return _macro_f1_codes(t, p, k), float(lo), float(hi)


def paired_bootstrap_diff(y_true, pred_a, pred_b, n=1000, alpha=0.05, seed=0, groups=None):
    """metric(B) - metric(A) on the SAME resamples. If the CI contains 0, it is not a claim."""
    labels = sorted(set(map(str, y_true)) | set(map(str, pred_a)) | set(map(str, pred_b)))
    idx = {l: i for i, l in enumerate(labels)}
    t, a, b = (np.array([idx[str(v)] for v in x]) for x in (y_true, pred_a, pred_b))
    k = len(labels)
    rng = np.random.default_rng(seed)
    draw = _resampler(len(t), groups, rng)
    d = np.empty(n)
    for i in range(n):
        ix = draw()
        d[i] = _macro_f1_codes(t[ix], b[ix], k) - _macro_f1_codes(t[ix], a[ix], k)
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return _macro_f1_codes(t, b, k) - _macro_f1_codes(t, a, k), float(lo), float(hi)


def sliced_report(df: pd.DataFrame, y_col: str, p_col: str, slices: list[str], min_n=50, n_boot=1000,
                  group_col: str | None = None, min_clusters=10) -> pd.DataFrame:
    """Per-slice macro-F1 with a CI. With group_col, CIs resample clusters and the cluster count is reported."""
    def grp(g):
        return None if group_col is None else g[group_col]

    def flag(g):
        notes = []
        if len(g) < min_n:
            notes.append(f"n<{min_n}: wide CI, not trusted")
        if group_col is not None and n_clusters(g[group_col]) < min_clusters:
            notes.append(f"{n_clusters(g[group_col])} clusters: CI is indicative only")
        return "; ".join(notes)

    rows = []
    f, lo, hi = bootstrap_ci(df[y_col], df[p_col], n_boot, groups=grp(df))
    rows.append(("ALL", "ALL", len(df), n_clusters(df[group_col]) if group_col else len(df), f, lo, hi, flag(df)))
    for col in slices:
        for val, g in df.groupby(col):
            k = n_clusters(g[group_col]) if group_col else len(g)
            if len(g) < 5 or k < 2:
                rows.append((col, val, len(g), k, np.nan, np.nan, np.nan, "too small"))
                continue
            f, lo, hi = bootstrap_ci(g[y_col], g[p_col], n_boot, groups=grp(g))
            rows.append((col, val, len(g), k, f, lo, hi, flag(g)))
    return pd.DataFrame(rows, columns=["slice", "value", "n", "clusters", "macro_f1", "ci_lo", "ci_hi", "flag"])


# ---- NER (strict, entity level, like seqeval's default) -----------------------------------
def _spans(tags):
    spans, start, typ = set(), None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if start is not None and (t == "O" or t.startswith("B-") or t[2:] != typ):
            spans.add((typ, start, i))
            start, typ = None, None
        if start is None and t != "O":  # a stray I- starts a span, like seqeval
            start, typ = i, t[2:]
    return spans


def entity_f1(true_seqs, pred_seqs) -> dict:
    tp = fp = fn = 0
    per = {}
    for ts, ps in zip(true_seqs, pred_seqs):
        T, P = _spans(ts), _spans(ps)
        tp += len(T & P)
        fp += len(P - T)
        fn += len(T - P)
        for typ in {s[0] for s in T | P}:
            t_, p_ = {s for s in T if s[0] == typ}, {s for s in P if s[0] == typ}
            d = per.setdefault(typ, [0, 0, 0])
            d[0] += len(t_ & p_)
            d[1] += len(p_ - t_)
            d[2] += len(t_ - p_)

    def f1(a, b, c):
        return 0.0 if a == 0 else 2 * a / (2 * a + b + c)

    return {"entity_f1": f1(tp, fp, fn), "per_type": {k: f1(*v) for k, v in sorted(per.items())}}


# ---- retrieval ----------------------------------------------------------------------------
def hit_at_k(ranks, k):
    return float(np.mean([r is not None and r <= k for r in ranks]))


def mrr_at_k(ranks, k):
    return float(np.mean([1.0 / r if (r is not None and r <= k) else 0.0 for r in ranks]))
