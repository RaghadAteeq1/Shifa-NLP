# Shifa Evaluation Report (test split)

Topic model served: `onnx-int8` | n = 210 messages in 36 wordings (clusters) | cluster bootstrap, 1000 resamples, 95% CI

## Targets

| Target | Status | Measured |
|---|---|---|
| Topic macro-F1 gain over TF-IDF >= +0.08 (paired CI lower bound > 0) | FAIL | -0.071 [-0.150, -0.018] |
| NER entity-F1 >= 0.80 | PASS | 0.962 |
| recall@10 >= 0.80 | PASS | 0.900 |
| MRR@10 >= 0.70 | PASS | 0.773 |
| Invariance >= 95% | FAIL | 0.941 |
| MFT >= 90% | PASS | 0.958 |
| Hand-read errors >= 100 | PASS | 104 |

## Overall topic macro-F1

- Model: **0.929** [0.850, 0.982] (item-level bootstrap would claim [0.890, 0.962])
- TF-IDF baseline: 1.000 [1.000, 1.000]
- Paired gain: **-0.071** [-0.150, -0.018]
- Gulf-slice paired gain: -0.115 [-0.324, +0.000]

## Sliced macro-F1 (model)

| slice | value | n | clusters | macro_f1 | ci_lo | ci_hi | flag |
|---|---|---|---|---|---|---|---|
| ALL | ALL | 210 | 36 | 0.929 | 0.850 | 0.982 |  |
| dialect | en | 63 | 12 | 0.950 | 0.707 | 1.000 |  |
| dialect | gulf | 84 | 12 | 0.885 | 0.676 | 1.000 |  |
| dialect | msa | 63 | 12 | 0.952 | 0.804 | 1.000 |  |
| length | long | 178 | 36 | 0.936 | 0.863 | 0.983 |  |
| length | short | 32 | 11 | 0.798 | 0.657 | 1.000 | n<50: wide CI, not trusted |
| topic | appointment | 53 | 9 | 0.961 | 0.850 | 1.000 | 9 clusters: CI is indicative only |
| topic | billing | 53 | 9 | 0.907 | 0.781 | 1.000 | 9 clusters: CI is indicative only |
| topic | lab_results | 52 | 9 | 0.980 | 0.950 | 1.000 | 9 clusters: CI is indicative only |
| topic | prescription | 52 | 9 | 1.000 | 1.000 | 1.000 | 9 clusters: CI is indicative only |

> Slices with n < 50 or fewer than 10 clusters are flagged, not hidden. With few wordings per slice the interval is wide: that is the honest size of the evidence.

## NER (strict entity level)

Entity-F1 **0.962**


| type | f1 |
|---|---|
| DATE | 0.976 |
| DEPT | 0.835 |
| DRUG | 1.000 |
| MONEY | 1.000 |
| PERSON | 1.000 |
| TEST | 1.000 |


## Semantic search

```json
{
  "hit@10": 0.9,
  "mrr@10": 0.773265306122449,
  "cross_hit@10": 0.41904761904761906,
  "cross_mrr@10": 0.1612849584278156,
  "hit@10[ar]": 0.9523809523809523,
  "cross_hit@10[ar\u2192other]": 0.30612244897959184,
  "hit@10[en]": 0.7777777777777778,
  "cross_hit@10[en\u2192other]": 0.6825396825396826
}
```

## Behavioural tests

- Invariance: **0.941** over 888 pairs; by kind: {"swap_date": 0.95, "swap_test": 0.865, "greeting": 0.957, "thanks": 0.957, "add_contact": 0.948, "swap_dept": 0.725, "swap_drug": 0.981, "swap_person": 1.0, "swap_amount": 1.0}
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
