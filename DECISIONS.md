# DECISIONS

Every model choice is argued from the architecture first, then backed by evidence **from our own runs**.
All numbers below come from our own runs on the **validation** split (`reports/eval_val.md`, `BENCHMARKS.md`, `reports/tokenizer_audit.csv`). Test numbers are in `EVALUATION_REPORT.md` and were produced once by `make final`.

## D0. Data
- 1,400 synthetic messages (no real patients, PDPL). Gulf 40% / MSA 30% / English 30%.
- Split by **template**: validation and test use wordings never seen in training, so a memorising model cannot score well. Patient IDs are also kept disjoint across splits, but the generator does not make text depend on the patient, so that split is bookkeeping: the template split is what prevents leakage.
- Test split frozen by SHA-256 (`data/test.sha256`); opening it is logged (`reports/test_access.log`).

## D1. Tokenizer  (#tokenizer)
- Question: which tokenizer should feed the classifier and NER?
- Architecture reason: WordPiece/SentencePiece vocabularies learnt mostly on English split Arabic words with clitics (و، ب، ال) into many pieces, which raises fertility, lengthens sequences, and spends attention (O(n²)) on fragments.
- Procedure: `make audit` -> `reports/tokenizer_audit.csv` (fertility, UNK rate, p95 length per language).
- Evidence (`reports/tokenizer_audit.csv`):

| tokenizer | ar fertility | en fertility | ar p95 len | en p95 len | UNK |
|---|---|---|---|---|---|
| distilbert-base-multilingual-cased | 2.29 | 1.36 | 35 | 27 | 0 |
| xlm-roberta-base | 1.76 | 1.26 | 26 | 26 | 0 |
| CAMeL-Lab/bert-base-arabic-camelbert-mix | **1.38** | 2.41 | 22 | 45 | 0 |
| aubmindlab/bert-base-arabertv2 | 1.68 | 1.98 | 27 | 38 | 0 |

- Reading: the Arabic-only vocabularies are best on Arabic but split English into many pieces (CAMeLBERT 2.41); our traffic is
  bilingual, so the balanced vocabulary is `xlm-roberta-base` (1.76 / 1.26). `distilbert-base-multilingual-cased` is the worst on Arabic
  (2.29), i.e. about 30% more fragments than XLM-R.
- Choice and trade-off: we kept DistilBERT (6 layers) because the p99 budget is the binding constraint: XLM-R base has 12 layers, so we expect
  roughly double the latency (an estimate, not measured), and the INT8 rung is already at 42.9 ms against a 40 ms budget. We did **not** train XLM-R end to end, so we cannot
  say what accuracy it would have bought. That is the first experiment to run on a faster machine (`SHIFA_CLASSIFIER=xlm-roberta-base`).
- `MAX_LEN = 128` is kept as a safe margin: the largest p95 in the audit is 45 tokens, far below the 512 positional limit.

## D2. Topic classifier checkpoint
- Encoder-only, because classification needs a bidirectional representation of the whole message ([CLS] pooled); a decoder adds cost without benefit here.
- Default `distilbert-base-multilingual-cased`: Arabic + English in one vocabulary, 6 layers, so it is cheap enough for the p99 budget. Alternatives are compared by D1 and by validation macro-F1.
- Evidence (validation, 210 messages, 36 wordings): TF-IDF baseline **0.805** [0.666, 0.920]; fp32 model 0.825 at the selected epoch;
  served INT8 model **0.814** [0.688, 0.915]. Paired gain of the served model over the baseline: **+0.009** [-0.141, +0.141].
  The target (+0.08 with a lower bound above 0) is **not met**.
- Why: with 8 training templates the TF-IDF baseline already reads most wordings (0.805; it was 0.626 when we had only 4 templates).
  The encoder helps on unseen phrasing, but the remaining errors are intent overlaps that neither model separates
  (appointment 0.74, billing 0.85; lab_results and prescription are at 0.99). See the error taxonomy in `EVALUATION_REPORT.md`.
- **Frozen test (opened once):** paired gain **-0.071** [-0.150, -0.018]. The baseline is better on test and the interval excludes zero,
  so the target fails in the wrong direction. We did not re-run or re-select after seeing it. Open question: whether the
  INT8 tax, the small number of training wordings, or the single validation template set used for checkpoint selection explains it.
  None of these was tested.
- We did not tune the data to make the baseline weaker. We report the gap as measured.

## D3. NER
- Token classification with the first-sub-token rule (other sub-tokens are -100) so one word = one tag.
- Strict entity-level F1 (a boundary miss scores zero).
- Kept in fp32 (see D5).
- Evidence (validation): entity-F1 **0.902** (target 0.80 met). Per type: DRUG 1.00, MONEY 1.00, PERSON 1.00, DATE 0.88, TEST 0.78, DEPT 0.73.
  Frozen test entity-F1: **0.962** (target met). The weakest validation types are the ones whose wording varies most across templates. In the first version (4 training templates) entity-F1 was 0.743
  and DEPT and TEST scored 0.0 because the model had learnt template positions instead of context.

## D4. Search
- Bi-encoder (`paraphrase-multilingual-MiniLM-L12-v2`) because it embeds each text independently, so the index is built once and queried in milliseconds.
- Cross-encoder re-ranker on the top-50 only, because it reads query and document together (full attention, more accurate, but cost grows with candidates).
- L2-normalised on **both** sides with inner product (cosine); the manifest pins embedder, corpus hash and preprocessing fingerprint.
- Evidence (validation): recall@10 **0.757** (target 0.80), MRR@10 **0.586** (target 0.70): **both not met**. By query language:
  Arabic 0.762, English 0.746.
- Cross-lingual gap: when the relevant case must be in the *other* language, hit@10 is 0.156 for Arabic queries and 0.603 for English queries
  (0.290 overall). Arabic to English retrieval is the weak direction.
- Frozen test: recall@10 **0.900**, MRR@10 **0.773**: both targets met. Validation was lower (0.757 / 0.586), so the score depends strongly on which held-out wordings are used.
- Not measured: we have no ablation of bi-encoder alone vs bi-encoder + cross-encoder, so we cannot say whether the re-ranker helps on short
  messages. That ablation is the next step, and it is the first thing to check before spending more on this stage.

## D5. Quantisation split
- Classifier: dynamic INT8 (weights int8). Hypothesis: small paired tax, large latency gain.
- NER: fp32. Hypothesis: boundary decisions are the most fragile outputs under quantisation.
- Evidence (`BENCHMARKS.md`, batch 1, CPU, 2 threads, n = 300; the reference rung for the paired tax is `2_torch_fp32_dynamic_pad`):

| Rung | p50 ms | p99 ms | macro-F1 | Paired tax [95% CI] |
|---|---|---|---|---|
| 1 torch fp32, pad to max | 126.4 | 571.1 | 0.825 | +0.000 |
| 2 torch fp32, dynamic pad | 51.7 | 237.8 | 0.825 | +0.000 |
| 3 ONNX fp32 | 26.5 | 85.2 | 0.825 | +0.000 |
| 4 ONNX INT8 | 15.0 | 42.9 | 0.814 | -0.011 [-0.069, +0.058] |

- Reading: ONNX alone gives a 2.8x cut in p99 at zero accuracy cost; INT8 gives another 2x for a tax of 0.011 whose interval includes 0,
  so it is within noise at this sample size (we do not claim it is free).
- The NER hypothesis (INT8 would hurt boundaries most) was **not tested**: NER is served in fp32 and we did not measure an INT8 NER.
- p99 at batch 1 is 42.9 ms against the 40 ms budget: just over.
- HTTP, 16 concurrent clients, 800 requests (`reports/http_load.json`): p50 270 ms, p99 **542 ms**, 56 requests/s. **Budget not met.**
  This run was on a free Colab VM with 2 vCPUs shared between the server and the load generator. Throughput of about 56 requests/s means each
  request waits for its turn (16 clients / 56 per s is about 285 ms, which is the observed p50). A p99 of 40 ms at 16 concurrent clients needs
  roughly 400 requests/s. Next steps: more cores (`make serve WORKERS=4`), a smaller encoder, or a GPU.
- Rollback: `artifacts/topic-onnx-fp32/` is kept; the service loads INT8 first and falls back by directory.

## D6. Dialect ID
- Char n-gram logistic regression, because the service needs it at startup without heavy dependencies.
- Limitation: on synthetic text it is near-perfect, which says nothing about real dialect traffic. For production use the CAMeL Tools identifier from Lab 4.

## D7. Confidence intervals resample wordings, not messages
- Question: how wide is the uncertainty on a split of 210 messages?
- Reason: validation and test each contain 36 wordings (4 topics x 3 dialects x 3 held-out templates). Messages inside a wording
  share every word except the slot values, so they are not independent. An item-level bootstrap treats them as 210
  independent samples and gives an interval that is far too narrow.
- Procedure: `shifa.metrics` resamples whole clusters (`groups=` argument, cluster id from `shifa.data.clusters`). Every
  CI and paired difference in the evaluation report and the ladder uses it, and the report prints the item-level interval
  next to it so the difference is visible. Slices with fewer than 10 wordings are flagged.
- Consequence: the first version had only 12 wordings per split (4 train templates, 1 val, 1 test): the model memorised the
  templates (train loss 0.21 while val loss rose) and the paired-gain CI was [-0.15, +0.36]. We widened the generator to 14
  templates per topic and dialect (8 train, 3 val, 3 test) rather than choosing a narrower statistic.
- Evidence (served model, validation macro-F1 0.814): cluster-bootstrap CI [0.688, 0.915] (width 0.227) against the item-level CI
  [0.767, 0.862] (width 0.095). Treating the 210 messages as independent would claim an interval 2.4 times too narrow.
  Slices of 12 wordings (one dialect) have intervals up to 0.50 wide, and the report flags them.
