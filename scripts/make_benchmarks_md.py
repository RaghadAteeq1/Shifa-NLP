"""Write BENCHMARKS.md from the JSON files our own runs produced (never typed by hand)."""
import json

from shifa import config as C


def load(name):
    p = C.REP / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ladder, thr, http = load("ladder.json"), load("throughput.json"), load("http_load.json")
    out = ["# BENCHMARKS (generated from our own runs)\n", "Re-create with `make bench bench-batch` and `make serve` + `make load`.\n"]
    if ladder:
        out += [f"## Optimisation ladder (batch size 1, CPU, {ladder['threads']} threads, n={ladder['n_run']})\n",
                f"Reference rung for the paired accuracy tax: `{ladder['reference']}`.\n",
                "| Rung | p50 ms | p99 ms | macro-F1 | Paired tax vs reference [95% CI] |", "|---|---|---|---|---|"]
        for r in ladder["rows"]:
            out.append(f"| {r['rung']} | {r['p50_ms']:.1f} | {r['p99_ms']:.1f} | {r['macro_f1']:.3f} | {r['tax_vs_ref']:+.3f} [{r['tax_ci'][0]:+.3f}, {r['tax_ci'][1]:+.3f}] |")
        out.append("\n![ladder](reports/fig_ladder.png)\n")
    if http:
        out += ["## HTTP latency of /v1/classify\n",
                f"- model: `{http['model']}`, concurrency {http['concurrency']}, {http['requests']} requests, {http['rps']:.0f} req/s",
                f"- p50 {http['p50_ms']:.1f} ms | p95 {http['p95_ms']:.1f} ms | **p99 {http['p99_ms']:.1f} ms** (budget {http['budget_ms']} ms): {'WITHIN' if http['within_budget'] else 'OVER'} budget\n"]
    if thr:
        out += [f"## /v1/classify:batch throughput (model `{thr['model']}`, {thr['n']} mixed-length messages)\n",
                "| Batch size | Fixed batches (items/s) | Length-bucketed (items/s) | Pad waste fixed | Pad waste bucketed |", "|---|---|---|---|---|"]
        for r in thr["rows"]:
            out.append(f"| {r['batch_size']} | {r['naive_items_per_sec']:.0f} | {r['bucketed_items_per_sec']:.0f} | {r['naive_pad_waste']:.2f} | {r['bucketed_pad_waste']:.2f} |")
        out.append("\n![throughput](reports/fig_throughput.png)\n")
    if len(out) == 2:
        out.append("No benchmark has been run yet.")
    (C.ROOT / "BENCHMARKS.md").write_text("\n".join(out), encoding="utf-8")
    print("BENCHMARKS.md written")


if __name__ == "__main__":
    main()
