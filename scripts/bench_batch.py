"""Extension: throughput vs batch size, naive fixed batches vs length-bucketed (feeds the /v1/classify:batch claim)."""
import json
import random
import time

from shifa import config as C
from shifa.batching import length_bucket_batches, padding_waste
from shifa.data import load_split
from shifa.engine import load_topic
from shifa.preprocess import preprocess

SIZES = [1, 2, 4, 8, 16, 32, 64]


def items_per_sec(fn, n, reps=3):
    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = max(best, n / (time.perf_counter() - t0))
    return best


def main():
    model, name = load_topic()
    va = load_split("val")
    # a mixed-length workload: short chat lines and long messages, shuffled like real traffic
    texts = list(va.text) * 2 + [t + " " + t + " " + t for t in va.text[:60]]
    random.Random(0).shuffle(texts)
    texts = texts[:256]
    lengths = [len(preprocess(t).split()) for t in texts]
    rows = []
    for bs in SIZES:
        naive = [list(range(i, min(i + bs, len(texts)))) for i in range(0, len(texts), bs)]
        bucket = length_bucket_batches(lengths, max_items=bs, max_tokens=10**9)

        def run(batches):
            def go():
                for b in batches:
                    model.predict_proba([texts[i] for i in b])
            return go
        rows.append({"batch_size": bs,
                     "naive_items_per_sec": items_per_sec(run(naive), len(texts)),
                     "bucketed_items_per_sec": items_per_sec(run(bucket), len(texts)),
                     "naive_pad_waste": padding_waste(lengths, naive),
                     "bucketed_pad_waste": padding_waste(lengths, bucket)})
        print(rows[-1])
    C.REP.mkdir(exist_ok=True)
    (C.REP / "throughput.json").write_text(json.dumps({"model": name, "n": len(texts), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
