"""The optimisation ladder (Lab 7 idea): every rung gets p50/p99 AND a paired accuracy tax. Own runs only.

Rungs: 1 torch fp32 padded to max_length -> 2 torch fp32 dynamic padding (reference)
       -> 3 ONNX fp32 -> 4 ONNX INT8 (dynamic). Single request, batch size 1, CPU.
"""
import json
import os
import time

import numpy as np

from shifa import config as C
from shifa import metrics as M
from shifa.data import clusters, load_split
from shifa.engine import TopicClassifier
from shifa.preprocess import preprocess

N_WARM, N_RUN = 30, 300


def timed(fn, texts):
    for t in texts[:N_WARM]:
        fn([t])
    lat, preds = [], []
    for t in texts[:N_RUN]:
        t0 = time.perf_counter()
        p = fn([t])
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append(p[0])
    return np.array(lat), preds


def torch_rungs(texts):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("SHIFA_THREADS", "2")))
    tok = AutoTokenizer.from_pretrained(str(C.TOPIC_FP32))
    model = AutoModelForSequenceClassification.from_pretrained(str(C.TOPIC_FP32)).eval()
    labels = C.TOPICS

    def run(padding):
        def fn(batch):
            enc = tok([preprocess(t) for t in batch], padding=padding, truncation=True, max_length=C.MAX_LEN, return_tensors="pt")
            with torch.no_grad():
                return [labels[i] for i in model(**enc).logits.argmax(-1).tolist()]
        return fn
    return {"1_torch_fp32_pad_to_max": run("max_length"), "2_torch_fp32_dynamic_pad": run(True)}


def main():
    va = load_split("val")
    texts, gold = list(va.text), list(va.topic)
    rungs = {}
    try:
        rungs.update(torch_rungs(texts))
    except Exception as e:
        print("torch rungs skipped:", e)
    for name, d in (("3_onnx_fp32", C.TOPIC_ONNX_FP32), ("4_onnx_int8", C.TOPIC_ONNX_INT8)):
        if (d / "model.onnx").exists():
            m = TopicClassifier(d)
            rungs[name] = m.predict

    rows, preds_by = [], {}
    for name, fn in rungs.items():
        lat, preds = timed(fn, texts)
        full = fn(texts)                                    # accuracy on the whole val set
        preds_by[name] = full
        f = M.macro_f1(gold, full)
        rows.append({"rung": name, "p50_ms": float(np.percentile(lat, 50)), "p99_ms": float(np.percentile(lat, 99)), "macro_f1": f})
    ref = "2_torch_fp32_dynamic_pad" if "2_torch_fp32_dynamic_pad" in preds_by else next(iter(preds_by))
    for r in rows:
        d, lo, hi = M.paired_bootstrap_diff(gold, preds_by[ref], preds_by[r["rung"]], groups=clusters(va))
        r.update({"tax_vs_ref": d, "tax_ci": [lo, hi]})
    C.REP.mkdir(exist_ok=True)
    (C.REP / "ladder.json").write_text(json.dumps({"reference": ref, "rows": rows, "n_run": N_RUN, "threads": os.environ.get("SHIFA_THREADS", "2")}, indent=2))
    print(f"{'rung':28s} {'p50':>8s} {'p99':>8s} {'F1':>7s} {'tax vs ref (95% CI)':>26s}")
    for r in rows:
        print(f"{r['rung']:28s} {r['p50_ms']:8.1f} {r['p99_ms']:8.1f} {r['macro_f1']:7.3f}   {r['tax_vs_ref']:+.3f} [{r['tax_ci'][0]:+.3f}, {r['tax_ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
