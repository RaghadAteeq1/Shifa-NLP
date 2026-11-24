# Shifa Evaluation Report (val split)

Topic model served: `onnx-int8` | n = 210 messages in 36 wordings (clusters) | cluster bootstrap, 1000 resamples, 95% CI

## Targets

| Target | Status | Measured |
|---|---|---|
| Topic macro-F1 gain over TF-IDF >= +0.08 (paired CI lower bound > 0) | FAIL | +0.009 [-0.141, +0.141] |
| NER entity-F1 >= 0.80 | PASS | 0.902 |
| recall@10 >= 0.80 | FAIL | 0.757 |
| MRR@10 >= 0.70 | FAIL | 0.586 |
| Invariance >= 95% | FAIL | 0.940 |
| MFT >= 90% | PASS | 0.958 |
| Hand-read errors >= 100 | PASS | 104 |

## Overall topic macro-F1

- Model: **0.814** [0.688, 0.915] (item-level bootstrap would claim [0.767, 0.862])
- TF-IDF baseline: 0.805 [0.666, 0.920]
- Paired gain: **+0.009** [-0.141, +0.141]
- Gulf-slice paired gain: -0.028 [-0.326, +0.274]

## Sliced macro-F1 (model)

| slice | value | n | clusters | macro_f1 | ci_lo | ci_hi | flag |
|---|---|---|---|---|---|---|---|
| ALL | ALL | 210 | 36 | 0.814 | 0.688 | 0.915 |  |
| dialect | en | 63 | 12 | 0.814 | 0.532 | 0.962 |  |
| dialect | gulf | 84 | 12 | 0.806 | 0.501 | 1.000 |  |
| dialect | msa | 63 | 12 | 0.815 | 0.593 | 1.000 |  |
| length | long | 192 | 35 | 0.822 | 0.674 | 0.922 |  |
| length | short | 18 | 6 | 0.500 | 0.500 | 1.000 | n<50: wide CI, not trusted; 6 clusters: CI is indicative only |
| topic | appointment | 53 | 9 | 0.738 | 0.455 | 0.928 | 9 clusters: CI is indicative only |
| topic | billing | 53 | 9 | 0.848 | 0.625 | 1.000 | 9 clusters: CI is indicative only |
| topic | lab_results | 52 | 9 | 0.990 | 0.958 | 1.000 | 9 clusters: CI is indicative only |
| topic | prescription | 52 | 9 | 0.990 | 0.969 | 1.000 | 9 clusters: CI is indicative only |

> Slices with n < 50 or fewer than 10 clusters are flagged, not hidden. With few wordings per slice the interval is wide: that is the honest size of the evidence.

## NER (strict entity level)

Entity-F1 **0.902**


| type | f1 |
|---|---|
| DATE | 0.877 |
| DEPT | 0.733 |
| DRUG | 1.000 |
| MONEY | 1.000 |
| PERSON | 1.000 |
| TEST | 0.785 |


## Semantic search

```json
{
  "hit@10": 0.7571428571428571,
  "mrr@10": 0.5858824640967498,
  "cross_hit@10": 0.2904761904761905,
  "cross_mrr@10": 0.15250944822373394,
  "hit@10[ar]": 0.7619047619047619,
  "cross_hit@10[ar\u2192other]": 0.1564625850340136,
  "hit@10[en]": 0.746031746031746,
  "cross_hit@10[en\u2192other]": 0.6031746031746031
}
```

## Behavioural tests

- Invariance: **0.940** over 838 pairs; by kind: {"greeting": 0.952, "thanks": 0.933, "add_contact": 0.948, "swap_amount": 1.0, "swap_date": 0.853, "swap_drug": 0.962, "swap_dept": 1.0, "swap_test": 0.968, "swap_person": 0.917}
- MFT: **0.958** over 24 items

## Error taxonomy (104 hand-read errors)

| category | share | count |
|---|---|---|
| similar_intent | 0.404 | 42 |
| perturbation_sensitive | 0.288 | 30 |
| short_text | 0.125 | 13 |
| entity_confusion | 0.087 | 9 |
| entity_boundary | 0.058 | 6 |
| dialect_wording | 0.038 | 4 |


## Limitations

Written by hand: see README (synthetic data, template-level shift, dialect ID is lexical).
