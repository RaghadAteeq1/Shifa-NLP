# DECISIONS

Every model choice is argued from the architecture first, then backed by evidence **from our own runs**.
Items marked `TODO(evidence)` must be filled from `reports/` before submission. Do not paste numbers from anywhere else.

## D0. Data
- 1,400 synthetic messages (no real patients, PDPL). Gulf 40% / MSA 30% / English 30%.
- Split by **template**: validation and test use wordings never seen in training, so a memorising model cannot score well. Patient IDs are also kept disjoint across splits, but the generator does not make text depend on the patient, so that split is bookkeeping: the template split is what prevents leakage.
- Test split frozen by SHA-256 (`data/test.sha256`); opening it is logged (`reports/test_access.log`).

## D1. Tokenizer  (#tokenizer)
- Question: which tokenizer should feed the classifier and NER?
- Architecture reason: WordPiece/SentencePiece vocabularies learnt mostly on English split Arabic words with clitics (و، ب، ال) into many pieces, which raises fertility, lengthens sequences, and spends attention (O(n²)) on fragments.
- Procedure: `make audit` -> `reports/tokenizer_audit.csv` (fertility, UNK rate, p95 length per language).
- Evidence: `TODO(evidence)` paste the table and the winner; set `MAX_LEN` from the p95 length (the positional table caps BERT-style models at 512).

## D2. Topic classifier checkpoint
- Encoder-only, because classification needs a bidirectional representation of the whole message ([CLS] pooled); a decoder adds cost without benefit here.
- Default `distilbert-base-multilingual-cased`: Arabic + English in one vocabulary, 6 layers, so it is cheap enough for the p99 budget. Alternatives are compared by D1 and by validation macro-F1.
- Evidence: `TODO(evidence)` baseline vs model macro-F1 with paired CI (`reports/eval_val.md`).

## D3. NER
- Token classification with the first-sub-token rule (other sub-tokens are -100) so one word = one tag.
- Strict entity-level F1 (a boundary miss scores zero).
- Kept in fp32 (see D5).

## D4. Search
- Bi-encoder (`paraphrase-multilingual-MiniLM-L12-v2`) because it embeds each text independently, so the index is built once and queried in milliseconds.
- Cross-encoder re-ranker on the top-50 only, because it reads query and document together (full attention, more accurate, but cost grows with candidates).
- L2-normalised on **both** sides with inner product (cosine); the manifest pins embedder, corpus hash and preprocessing fingerprint.
- Evidence: `TODO(evidence)` recall@10, MRR@10 and the cross-lingual gap from the evaluation report.

## D5. Quantisation split
- Classifier: dynamic INT8 (weights int8). Hypothesis: small paired tax, large latency gain.
- NER: fp32. Hypothesis: boundary decisions are the most fragile outputs under quantisation.
- Evidence: `TODO(evidence)` the ladder table in BENCHMARKS.md with paired taxes. If the NER tax is within noise, say so and revisit.
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
- Evidence: `TODO(evidence)` cluster vs item-level CI from `reports/eval_val.md`.
