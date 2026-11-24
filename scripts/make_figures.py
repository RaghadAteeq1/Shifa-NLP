"""Charts for the README, drawn from the JSON/CSV the other scripts wrote (nothing hand-typed)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shifa import config as C

TEAL, GREY, RED = "#0f766e", "#7b8f8c", "#b42347"


def slices(split):
    p = C.REP / f"slices_{split}.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df.slice.isin(["ALL", "dialect", "length"]) & df.macro_f1.notna()].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = range(len(df))
    ax.errorbar(df.macro_f1, y, xerr=[df.macro_f1 - df.ci_lo, df.ci_hi - df.macro_f1], fmt="o", color=TEAL, capsize=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{s}: {v} (n={n})" for s, v, n in zip(df.slice, df.value, df.n)])
    ax.set_xlabel("macro-F1 with 95% bootstrap CI")
    ax.set_title("Topic macro-F1 per slice")
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout()
    fig.savefig(C.REP / f"fig_slices_{split}.png", dpi=160)


def ladder():
    p = C.REP / "ladder.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = [r["rung"] for r in rows]
    ax.bar(names, [r["p99_ms"] for r in rows], color=[TEAL if r["p99_ms"] <= 40 else GREY for r in rows])
    ax.axhline(40, color=RED, ls="--", label="budget 40 ms")
    for i, r in enumerate(rows):
        ax.text(i, r["p99_ms"] + 0.5, f"{r['p99_ms']:.1f}\ntax {r['tax_vs_ref']:+.3f}", ha="center", fontsize=8)
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("Optimisation ladder")
    ax.legend()
    plt.xticks(rotation=15, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.REP / "fig_ladder.png", dpi=160)


def throughput():
    p = C.REP / "throughput.json"
    if not p.exists():
        return
    rows = json.loads(p.read_text())["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bs = [r["batch_size"] for r in rows]
    ax.plot(bs, [r["naive_items_per_sec"] for r in rows], "o-", color=GREY, label="fixed batches")
    ax.plot(bs, [r["bucketed_items_per_sec"] for r in rows], "o-", color=TEAL, label="length-bucketed")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch size")
    ax.set_ylabel("items / second")
    ax.set_title("/v1/classify:batch throughput")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.REP / "fig_throughput.png", dpi=160)


if __name__ == "__main__":
    for s in ("val", "test"):
        slices(s)
    ladder()
    throughput()
    print("figures written to reports/")
