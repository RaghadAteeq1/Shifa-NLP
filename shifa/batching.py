"""Length-bucketed batching for /v1/classify:batch. Pure Python so it is unit-tested."""


def length_bucket_batches(lengths, max_items=32, max_tokens=2048):
    """Group indices of similar length so padding waste stays small.

    A batch's cost is (items x longest item), so it is closed when either limit is hit.
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and (len(cur) + 1 > max_items or new_max * (len(cur) + 1) > max_tokens):
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def run_bucketed(fn, items, lengths, max_items=32, max_tokens=2048):
    """fn(list_of_items) -> list of outputs. Returns outputs in the ORIGINAL order."""
    out = [None] * len(items)
    for batch in length_bucket_batches(lengths, max_items, max_tokens):
        res = fn([items[i] for i in batch])
        for i, r in zip(batch, res):
            out[i] = r
    return out


def padding_waste(lengths, batches):
    """Fraction of token slots that are padding. Lower is better."""
    total = sum(max(lengths[i] for i in b) * len(b) for b in batches)
    real = sum(lengths[i] for b in batches for i in b)
    return 0.0 if total == 0 else 1 - real / total
