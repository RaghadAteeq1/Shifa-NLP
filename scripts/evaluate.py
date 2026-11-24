"""Evaluation report (Lab 6): slices + bootstrap CIs, paired gain over baseline, NER, search, behaviour, taxonomy.

  python -m scripts.evaluate               -> validation split (free to use while developing)
  SHIFA_FINAL=1 python -m scripts.evaluate --split test   -> frozen test, writes EVALUATION_REPORT.md
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from shifa import config as C
from shifa import metrics as M
from shifa import behaviour
from shifa.data import clusters, load_split
from shifa.engine import load_ner, load_topic
from shifa.preprocess import preprocess


def md(df, floatfmt=3):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.{floatfmt}f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if (isinstance(v, float) and np.isnan(v)) else str(v)) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def search_eval(df, searcher):
    ranks_any, ranks_x, rows = [], [], []
    for r in df.itertuples():
        hits = searcher.search(r.text, k=50, n_candidates=100)
        qlang = "en" if r.dialect == "en" else "ar"

        def rank(cross):
            for i, h in enumerate(hits, 1):
                dlang = "en" if h["dialect"] == "en" else "ar"
                if h["topic"] == r.topic and h["intent"] == r.intent and (not cross or dlang != qlang):
                    return i
            return None
        ranks_any.append(rank(False))
        ranks_x.append(rank(True))
        rows.append(qlang)
    out = {"hit@10": M.hit_at_k(ranks_any, 10), "mrr@10": M.mrr_at_k(ranks_any, 10),
           "cross_hit@10": M.hit_at_k(ranks_x, 10), "cross_mrr@10": M.mrr_at_k(ranks_x, 10)}
    for lang in ("ar", "en"):
        idx = [i for i, l in enumerate(rows) if l == lang]
        out[f"hit@10[{lang}]"] = M.hit_at_k([ranks_any[i] for i in idx], 10)
        out[f"cross_hit@10[{lang}→other]"] = M.hit_at_k([ranks_x[i] for i in idx], 10)
    return out


def main(split):
    df = load_split(split).reset_index(drop=True)
    topic, name = load_topic()
    base = joblib.load(C.BASELINE)
    df["pred"] = topic.predict(list(df.text))
    df["base"] = list(base.predict([preprocess(t) for t in df.text]))
    df["length"] = np.where(df.n_words <= C.SHORT_WORDS, "short", "long")
    df["cluster"] = clusters(df)          # CIs resample wordings, not messages (DECISIONS.md D7)
    G = df.cluster

    rep = {"split": split, "topic_model": name, "n": len(df), "clusters": M.n_clusters(G)}
    f, lo, hi = M.bootstrap_ci(df.topic, df.pred, groups=G)
    _, lo_iid, hi_iid = M.bootstrap_ci(df.topic, df.pred)
    bf, blo, bhi = M.bootstrap_ci(df.topic, df.base, groups=G)
    gain = M.paired_bootstrap_diff(df.topic, df.base, df.pred, groups=G)
    rep["overall"] = {"model": [f, lo, hi], "model_iid_ci": [lo_iid, hi_iid], "baseline": [bf, blo, bhi],
                      "gain_vs_baseline": list(gain)}
    sl = M.sliced_report(df, "topic", "pred", ["dialect", "length", "topic"], group_col="cluster")
    sl_b = M.sliced_report(df, "topic", "base", ["dialect"], group_col="cluster")
    gulf = df[df.dialect == "gulf"]
    rep["gulf_gain"] = list(M.paired_bootstrap_diff(gulf.topic, gulf.base, gulf.pred, groups=gulf.cluster))

    ner = load_ner()
    if ner:
        pred_tags = [ner.tag_words([list(w)])[0] for w in df.words]
        rep["ner"] = M.entity_f1([list(t) for t in df.tags], pred_tags)
    try:
        from shifa.search import Searcher
        rep["search"] = search_eval(df, Searcher())
    except Exception as e:
        rep["search"] = {"error": str(e)}

    beh = behaviour.run_suite(topic.predict, df)
    rep["behaviour"] = {k: v for k, v in beh.items() if not k.endswith("failures")}

    # error taxonomy input: validation errors + behavioural failures. Test errors are NOT dumped.
    if split == "val":
        err = df[df.pred != df.topic][["id", "dialect", "topic", "pred", "text"]].copy()
        err["source"] = "validation"
        extra = pd.DataFrame([{"id": f"beh-{i}", "dialect": "", "topic": "", "pred": f"{y}->{z}", "text": o, "source": f"inv:{k}"}
                              for i, (k, o, n, y, z) in enumerate(beh["invariance_failures"][:60])])
        extra2 = pd.DataFrame([{"id": f"mft-{i}", "dialect": "", "topic": y, "pred": p, "text": t, "source": "mft"}
                               for i, (t, y, p) in enumerate(beh["mft_failures"])])
        parts_ = [err, extra, extra2]
        if len(err) + len(extra) + len(extra2) < 100:   # R4 needs >= 100 hand-read errors
            be = df[df.base != df.topic][["id", "dialect", "topic", "base", "text"]].rename(columns={"base": "pred"})
            be["source"] = "baseline-validation"
            parts_.append(be)
        out = pd.concat(parts_, ignore_index=True)
        out["category"] = ""
        C.REP.mkdir(exist_ok=True)
        out.to_csv(C.REP / "errors_to_read.csv", index=False, encoding="utf-8-sig")
        rep["errors_to_read"] = len(out)
    tagged = C.REP / "errors_tagged.csv"
    tax = None
    if tagged.exists():
        t = pd.read_csv(tagged)
        t = t[t.category.notna() & (t.category.astype(str).str.strip() != "")]
        tax = (t.category.value_counts() / max(1, len(t))).rename("share").reset_index()
        tax["count"] = t.category.value_counts().values
        rep["errors_read"] = int(len(t))

    # ---- write ----
    C.REP.mkdir(exist_ok=True)
    (C.REP / f"metrics_{split}.json").write_text(json.dumps(rep, indent=2, default=float))
    sl.to_csv(C.REP / f"slices_{split}.csv", index=False)
    sl_b.to_csv(C.REP / f"slices_baseline_{split}.csv", index=False)

    checks = [("Topic macro-F1 gain over TF-IDF >= +0.08 (paired CI lower bound > 0)", gain[0] >= 0.08 and gain[1] > 0, f"{gain[0]:+.3f} [{gain[1]:+.3f}, {gain[2]:+.3f}]")]
    if "ner" in rep:
        checks.append(("NER entity-F1 >= 0.80", rep["ner"]["entity_f1"] >= 0.80, f"{rep['ner']['entity_f1']:.3f}"))
    s = rep["search"]
    if "hit@10" in s:
        checks += [("recall@10 >= 0.80", s["hit@10"] >= 0.80, f"{s['hit@10']:.3f}"), ("MRR@10 >= 0.70", s["mrr@10"] >= 0.70, f"{s['mrr@10']:.3f}")]
    b = rep["behaviour"]
    checks += [("Invariance >= 95%", b["invariance_rate"] >= 0.95, f"{b['invariance_rate']:.3f}"), ("MFT >= 90%", b["mft_rate"] >= 0.90, f"{b['mft_rate']:.3f}")]
    if "errors_read" in rep:
        checks.append(("Hand-read errors >= 100", rep["errors_read"] >= 100, str(rep["errors_read"])))
    ck = pd.DataFrame([(c, "PASS" if ok else "FAIL", v) for c, ok, v in checks], columns=["Target", "Status", "Measured"])

    parts = [f"# Shifa Evaluation Report ({split} split)\n",
             f"Topic model served: `{name}` | n = {len(df)} messages in {rep['clusters']} wordings (clusters) | "
             "cluster bootstrap, 1000 resamples, 95% CI\n",
             "## Targets\n", md(ck), "\n## Overall topic macro-F1\n",
             f"- Model: **{f:.3f}** [{lo:.3f}, {hi:.3f}] (item-level bootstrap would claim [{lo_iid:.3f}, {hi_iid:.3f}])\n"
             f"- TF-IDF baseline: {bf:.3f} [{blo:.3f}, {bhi:.3f}]\n"
             f"- Paired gain: **{gain[0]:+.3f}** [{gain[1]:+.3f}, {gain[2]:+.3f}]\n"
             f"- Gulf-slice paired gain: {rep['gulf_gain'][0]:+.3f} [{rep['gulf_gain'][1]:+.3f}, {rep['gulf_gain'][2]:+.3f}]\n",
             "## Sliced macro-F1 (model)\n", md(sl), "\n> Slices with n < 50 or fewer than 10 clusters are flagged, not hidden. "
             "With few wordings per slice the interval is wide: that is the honest size of the evidence.\n"]
    if "ner" in rep:
        parts += ["## NER (strict entity level)\n", f"Entity-F1 **{rep['ner']['entity_f1']:.3f}**\n\n", md(pd.DataFrame(rep["ner"]["per_type"].items(), columns=["type", "f1"])), "\n"]
    parts += ["## Semantic search\n", "```json\n" + json.dumps(rep["search"], indent=2) + "\n```\n"]
    parts += ["## Behavioural tests\n", f"- Invariance: **{b['invariance_rate']:.3f}** over {b['invariance_n']} pairs; by kind: {json.dumps({k: round(v, 3) for k, v in b['invariance_by_kind'].items()})}\n- MFT: **{b['mft_rate']:.3f}** over {b['mft_n']} items\n"]
    if tax is not None:
        parts += [f"## Error taxonomy ({rep['errors_read']} hand-read errors)\n", md(tax), "\n"]
    else:
        parts += ["## Error taxonomy\n", "Not tagged yet: read `reports/errors_to_read.csv`, fill the `category` column, save as `reports/errors_tagged.csv`, re-run.\n"]
    parts += ["## Limitations\n", "Written by hand: see README (synthetic data, template-level shift, dialect ID is lexical).\n"]
    target = C.ROOT / "EVALUATION_REPORT.md" if split == "test" else C.REP / "eval_val.md"
    target.write_text("\n".join(parts), encoding="utf-8")
    print(md(ck))
    print("written:", target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    main(ap.parse_args().split)
