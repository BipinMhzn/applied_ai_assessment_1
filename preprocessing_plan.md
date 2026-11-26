# Data Preprocessing Plan — Stack Overflow Similar-Question Recommender

**Product (agreed scope):** *"StackLearn"* — a Streamlit app where the user pastes an AI/ML question
(title + body + optional code, optional domain tag) and gets the **top-k most similar
already-answered** Stack Overflow questions (title, tags, score, answer count, link). Content-based
retrieval over a catalog restricted to `is_answered == True`; no user accounts, no feedback loop.
Positioning sentence for the report: *"given a new AI/ML question (title, error text, code), it
recommends the most similar solved Stack Overflow questions so the developer can reuse existing
solutions."*

**Files:** `data_collection.ipynb` → `dataset/questions.csv` (raw) → **`data_preprocessing.ipynb`**
→ `dataset/questions_clean.csv` (clean catalog) → **`train_model.ipynb`** → model artifacts in
`outputs/` → **`app.py`** (Streamlit). One job per file.
**Input:** `dataset/questions.csv` (single raw CSV produced by `data_collection.ipynb` — see §1)
**Output:** `dataset/questions_clean.csv` + the runtime artifacts in `outputs/` that the
Streamlit app loads at startup (see §5 for exactly which files are git-tracked)
**Assessment mapping (CS5079 CA1 Task 1):** this document covers the *preprocessing / feature
engineering* part of the brief; §4 covers model choice, training & evaluation; §5 covers
deployment; the collection in §1 covers *data collection* and *data sufficiency*.

---

## 1. Data scope (collection) — decision: tag-scoped, multi-year

Collect **all** Stack Overflow questions over 2024-01-01 → 2026-12-31 (one pass), then filter to a
curated list of 26 AI/ML subject tags **in preprocessing** (Phase 1, runs first). Rationale: a
coherent AI/ML domain fits the CS5079 Applied AI theme, makes recommendations visibly better, and
makes tag-overlap a clean relevance signal for evaluation; collecting broadly first keeps the
collector code simple and still yields a large, diverse dataset (26 subjects × 3 years, including
the high-volume LLM-era tags of 2024–2026).

Tag list (`TAGS` in `data_collection.ipynb`, applied as the domain filter in Phase 1; meta-tags
like `advice`, `best-practices` deliberately excluded):
`python, machine-learning, deep-learning, neural-network, artificial-intelligence, data-science,
nlp, computer-vision, reinforcement-learning, tensorflow, keras, pytorch, scikit-learn, xgboost,
transformers, huggingface, langchain, openai-api, llm, prompt-engineering, spacy, nltk, opencv,
pandas, numpy, jupyter`

**Collection (in `data_collection.ipynb` — your original `collect()` is unchanged):** one pass
over 2024-01-01 → 2026-12-31 into a single `dataset/questions.csv` (all tags; the seen-ID set
dedupes at write time and makes the run resumable). The API's `tagged` parameter is AND-only, so it
cannot express the union of 26 tags in one request — the AI/ML filter therefore lives in
preprocessing (Phase 1), not in the collector. API quota with key: 10,000
requests/day (100 questions each). **Current full collection:** `dataset/questions.csv` =
528,495 unique questions (2024–2026 crawl of 527,695 plus ~800 canonical duplicate targets back
to 2008; profiled 2026-11-25). `collect()` is **resilient and resume-aware**: HTTP 429 (gateway rate limit) stops the run cleanly
with a clear message instead of hammering; other HTTP errors are retried up to 3 times per page
then the rest of that day is skipped with the status/error visible in the log; API `backoff` fields
are honored; daily quota exhaustion stops cleanly. The run cell starts from the newest
`creation_date` already in the CSV (not from `START_DATE`; seen IDs prevent duplicates), so an
error never forces a restart from 2024.

**Enrichment pass (one new, simple cell in the notebook):** `fetch_canonical_targets()` reads every
`duplicate_of` ID from the CSV and fetches those canonical questions **by ID** via
`/questions/{ids}` (batches of 100, any date), because canonicals are usually *older* than the
crawl range (in the 2026 sample only 19/1,040 targets were in-range). Only canonicals with ≥ 1 tag
in `TAGS` are kept, so the duplicate-redirect resolves to an in-catalog, usually accepted-answered
question. Answer bodies are deliberately **not** collected — results link to Stack Overflow
(scope guard).

### Baseline profile of the current CSV (measured, all-tags 2026 sample)

| Property | Value |
|---|---|
| Rows | 20,225 questions |
| Columns | `question_id, title, body_text, code_text, tags, score, view_count, answer_count, is_answered, has_accepted_answer, creation_date, closed_reason, duplicate_of` |
| Duplicate `question_id` | 0 |
| Date range | 2026-01-01 → 2026-11-15 (collection stopped mid-November) |
| Rows per month | Jan 3,586 → Nov 520 (declining — quota/time limited) |
| Empty `body_text` / `code_text` | 4 / 5,416 (26.8%); 0 rows with both empty |
| Closed questions | 2,188 (812 `Duplicate`; only 19 of 1,040 `duplicate_of` targets exist in-dataset) |
| `is_answered` / `has_accepted_answer` | 9,325 / 5,128 |
| `score` | min −15, median 0, p99 8, max 180 (answered mean 1.06 vs unanswered 0.05) |
| `view_count` | min 4, median 134, p99 1,241, max 26,040 |
| Tags | 1–5 per question, all lowercase, **10,047 unique**, 9,184 with < 10 occurrences |
| Text hygiene | 1,131 rows with `<...>` (mostly code-in-prose like `FC<T>`, **not** HTML), 1,731 with URLs, 13 markdown images, 10 with HTML entities, 3,165 with non-ASCII |
| Text lengths | title 15–150 chars; body median 671 / max 21,362; code median 419 / max 28,992 |

After tag-scoped multi-year collection, expect the long tail to shrink sharply (every row carries
≥ 1 curated tag) and row count to grow well beyond 20k — both strengthen the "large and diverse
data" justification required by the brief.

---

## 1b. Full-data run results (measured by running `data_preprocessing.ipynb` on the complete crawl)

| Stage | Rows |
|---|---|
| Loaded (`dataset/questions.csv`) | 528,495 |
| After AI/ML domain filter (≥ 1 tag in `TAGS`) — runs **first** | **74,126** (only ~14% of all SO 2024–2026 questions are AI/ML — the domain filter is what makes the product coherent, and running it first makes every later step ~7× cheaper) |
| After dedup / missing / closed-duplicate / off-topic removal | 69,880 (3,082 Duplicate-closed dropped, redirect map kept; 14 non-English dropped) |
| After quality filters + `catalog_answered_only` | **36,060** (12.5% of kept rows have no code — kept, not dropped) |
| Saved | `dataset/questions_clean.csv` (36,060 rows, ~143 MB) |

Top domain tags after the filter: `python` 67,964, `pandas` 7,064, `pytorch` 2,276,
`machine-learning` 1,908, `tensorflow` 1,853. Canonical enrichment extends the catalog back to
2008. `train_model.ipynb` then splits temporally (train ≤ 2026-08-31, val Sep–Oct, test Nov+)
and evaluates TF-IDF vs embeddings vs hybrid on the test split (see §4; current
`outputs/model_selection.json`: embeddings precision@5 = 0.990 vs TF-IDF 0.981, hybrid picked
w = 0.0 i.e. pure embeddings).

---

## 2. Pipeline overview

```
raw CSV → load & type-cast → AI/ML tag filter (FIRST) → integrity & dedup →
missing values → closed/duplicate policy → text cleaning → language filter →
quality filters → feature engineering → save dataset/questions_clean.csv
        │
        ▼  train_model.ipynb
temporal split → hyperparameter tuning on val → final TF-IDF fit with the BEST params →
test evaluation (TF-IDF vs embeddings) → persist model artifacts to outputs/
        │
        ▼  app.py (Streamlit)
```

Two notebooks, one job each: **`data_preprocessing.ipynb`** = preprocessing only (driven by its
`CONFIG` dict, writes `dataset/questions_clean.csv` + small sidecars in `outputs/`);
**`train_model.ipynb`** = load clean CSV → split → train/evaluate candidates → write model
artifacts to `outputs/`. Deterministic (fixed seeds, sorted vocab).

The AI/ML tag filter runs **first**, right after load: the raw crawl is all-tags and huge
(~528k rows), so filtering before the slow text steps (dedup, cleaning, language detection)
shrinks the working set to the AI/ML subset (~36k rows) and makes preprocessing fast.

---

## 3. Phase details

### Phase 0 — Load
- Single source: `pd.read_csv(CONFIG["src_path"])` where `src_path = "dataset/questions.csv"`;
  type-cast columns; dedup by `question_id` (collection already prevents duplicates, this is a guard).
- Cast: `creation_date` → datetime UTC; `is_answered`, `has_accepted_answer` → bool;
  `score`, `view_count`, `answer_count` → int; `question_id` → str.
- Parse `tags` (`"a|b|c"` → list), `duplicate_of` → list.
- **Validation:** schema match; row count logged; no nulls in key columns.

### Phase 1 — AI/ML domain filter (runs FIRST, before any text processing)
- Split `tags` on `|` (verified lowercase).
- **Domain filter:** keep only rows with ≥ 1 tag in `TAGS` (the collected AI/ML list, config
  `domain_tags`) — this is where the dataset is scoped to AI/ML. Log rows dropped.
- Runs immediately after load so the slow steps below work on the small AI/ML subset only
  (528k → 74k rows before any text cleaning runs).
- Tag vocabulary (`tag_vocab.json`) + multi-hot label matrix `Y` are built in
  **train_model.ipynb**, not here.
- **Validation:** row count before/after; print top-30 tags.

### Phase 2 — Integrity & dedup
- `question_id` must be unique (the collector's seen-ID set guarantees this at write time; still
  assert it as a guard for future collection runs).
- Exact-duplicate detection on normalized `title + body_text` hash.
- **Output:** deduplicated frame + `dropped_rows` log.

### Phase 3 — Missing values
- **`code_text` empty does NOT make a question invalid** — text-only questions are kept
  (12.5% of the AI/ML rows have no code); only a derived `has_code` flag is added.
- `body_text` empty → drop only if `code_text` also empty (0 such rows in the full crawl).
- `closed_reason` / `duplicate_of` empty → `""` / `[]` (structural, not missing).
- **Validation:** zero NaNs; counts logged.

### Phase 4 — Closed & duplicate policy
- **Drop** `Duplicate`-closed rows from the candidate pool, but **keep the `duplicate_of` map** as
  a redirect: if a recommended question is a duplicate, link to the canonical one. The canonical
  targets are fetched by the enrichment pass (§1), so most redirects resolve inside the catalog.
- **Drop** off-topic closed (`Not suitable`, `Opinion-based`); keep `Needs details/focus` but
  flag them. Config flag `drop_offtopic_closed`.
- **Validation:** rows removed per reason, stored in the drop log.

### Phase 5 — Text cleaning (per field; code treated gently)
- Unescape HTML entities (`&amp;` → `&`).
- URLs → `<URL>` placeholder; markdown images → `<IMG>`.
- Collapse whitespace; unicode-normalize (NFKC) but **keep non-ASCII**.
- **Do not** blanket-strip `<...>` — most are code-in-prose (`FC<T>`, `str[]`). Only remove
  well-formed HTML tags on known tag names.
- Keep original text for display; add `text_lower` for modeling.
- **Validation:** 20-row before/after sample; no entities remain; length distributions sane.

### Phase 6 — Language filter
- `langdetect` on `title + body_text`; keep English (`english_only: true`).
- **Validation:** language histogram in log.

### Phase 7 — Quality filters (config thresholds)
- `min_body_chars: 50`, `min_view_count: 5`.
- **Catalog restriction (the one scope drill-down):** only `is_answered == True` questions are
  served as recommendations (`catalog_answered_only: true`) — the product promise is "similar
  *solved* questions". Unanswered questions can still be kept in the training/eval split for
  coverage; `has_accepted_answer` is used to *rank* results (accepted-answer questions first),
  not to filter.
- **Validation:** rows dropped per rule; distributions before/after.

### Phase 8 — Feature engineering
- **Text:** `text_all = title×2 + " " + body_text + " " + code_text + " " + " ".join(tags)`
  (title weighted ×2 — configurable).
- **Numeric:** `log1p(score)`, `log1p(view_count)`, `answer_count`.
- **Flags:** `is_answered`, `has_accepted_answer`, `has_code`, `is_closed`, `is_duplicate`.
- **Temporal:** `year_month`, `day_of_week`, `hour`, `days_since_start`.
- **Text stats:** `title_len`, `body_len`, `code_len`, `code_ratio`.
- **Validation:** no inf/NaN; sanity-check correlations.

### Phase 9 — Save clean dataset + sidecars  *(end of data_preprocessing.ipynb)*
- `df.to_csv("dataset/questions_clean.csv")` — the single input for `train_model.ipynb` and `app.py`.
- `outputs/duplicate_map.csv`, `outputs/config.json`, `outputs/preprocessing_log.txt`.

### Phases 10–12 — moved to `train_model.ipynb`
The split / vectorization / candidate models / evaluation / persistence phases now live in
**train_model.ipynb**, which loads `dataset/questions_clean.csv` + `outputs/config.json`
(written by data_preprocessing.ipynb) and saves model artifacts to `outputs/`. See Phases 10–12 below.

### Phase 10 — Train/val/test split (temporal, leak-free)  *(in train_model.ipynb)*
Split **by `creation_date`**, not randomly (data is time-ordered; avoids leakage):
- Train: `creation_date <= 2026-08-31` · Val: Sep–Oct 2026 · Test: Nov 2026+ (`split_dates` in config)
- **Validation:** max(train) < min(val) < min(test); no `question_id` overlap across splits.

### Phase 11 — Tuning → final fit → candidates  *(in train_model.ipynb)*
Order matters: **tune first, then fit the final model with the winning params, then evaluate
on test** (never refit on val/test).
1. **Hyperparameter tuning (val only):** small grid over TF-IDF settings (word n-grams,
   char n-grams, `min_df`, `max_features`, stopwords), each fitted on train and scored by
   precision@5 on val. Winner: `best_tfidf_params`.
2. **Candidate A — TF-IDF, final:** `FeatureUnion` of word (1–2) + char_wb (3–5) TF-IDF built
   with `best_tfidf_params`, fit on **train only**, then transform val/test. Handles code
   tokens; no GPU; tiny artifact. The winning params are written back into `config.json`.
3. **Candidate B — sentence embeddings:** `all-MiniLM-L6-v2` (sentence-transformers) used
   **pretrained, off-the-shelf** — no tuning needed (nothing is trained; swapping checkpoints
   would be model selection, out of scope). Encodes catalog + queries; retrieval by cosine.
   Also encodes the full catalog (`E_all`) for the app.
4. **Test evaluation** of both candidates (Phase 5 protocol below); winner by precision@5 goes
   to `model_selection.json` and is what `app.py` serves.
- A **tag-only Jaccard baseline** and a **hybrid** (weighted TF-IDF + embeddings) were tried and
  removed: the baseline is circular (tag overlap is also the relevance metric) and the hybrid
  picked `w = 0.0` on val — pure embeddings, no gain — so neither earns its code.
- All candidates scored with the same protocol: queries = first `max_eval_queries` test rows,
  catalog = train, metrics precision@k / recall@k / MRR (k = 5, 10, tag-overlap relevance).
  Results collected in `MODEL_RESULTS`; the winner by precision@5 is recorded in
  `model_selection.json` and **that** model is what `app.py` loads and serves.
- Persist the fitted vectorizer with `joblib`; **never refit on val/test**.
- **Validation:** vocab size, sparsity, top-5 sanity check for 3 known queries.

### Phase 12 — Persist model artifacts  *(end of train_model.ipynb)*
```
dataset/
  questions_clean.csv         # TRACKED in git (un-ignored) — written by data_preprocessing.ipynb,
                              # loaded by train_model.ipynb AND app.py
outputs/
  vectorizer.joblib           # TRACKED — fitted TF-IDF vectorizer
  duplicate_map.csv           # TRACKED — closed-duplicate redirects (from data_preprocessing.ipynb)
  model_selection.json        # TRACKED — emb_model name + MODEL_RESULTS table (winner = best precision@5)
  embeddings_all.npz          # TRACKED — E_all: precomputed catalog embeddings (candidate B/C)
  config.json, tag_vocab.json, X_{train,val,test}.npz, Y.npz,
  preprocessing_log.txt, training_log.txt   # local only (gitignored, regenerable)
```
`.gitignore` keeps `dataset/questions.csv` (1.1 GB raw) and everything in `outputs/` local
**except** the four runtime artifacts above.

---

## 4. Evaluation plan (feeds the report's "validate effectiveness" requirement)

- **Quantitative:** on the temporal test split, for each test question retrieve top-k
  (k = 5, 10) and score with **precision@k / recall@k using tag overlap as the relevance proxy**
  (predicted question relevant if it shares ≥ 1 curated tag with the query), plus MRR.
  Report the candidates side by side: **TF-IDF vs sentence embeddings**, with the val grid
  search as the tuning evidence. `model_selection.json` holds the table; the report quotes it and
  argues the deployment choice (e.g. TF-IDF if it wins or ties — much smaller artifact,
  no torch/sentence-transformers download, faster cold start on Streamlit Cloud).
- **Metric caveat to disclose in the report:** with "shares ≥ 1 domain tag" as relevance,
  `python` alone matches most of the catalog, so precision@k is high and recall@k is tiny
  (relevant set ≈ 30k of 36k). For a sharper signal, also report a stricter proxy — e.g. shares
  a tag **other than `python`**, or tag-Jaccard > 0 — on the same run.
- **Qualitative:** 3–5 example queries with top-5 results shown in the report (screenshots) —
  markers reward visible, sensible recommendations.
- **Ablation (optional, cheap):** with/without `code_text` in `text_all` to show char n-grams help.

---

## 5. Deployment notes (Streamlit)

- App loads `dataset/questions_clean.csv` (written by data_preprocessing.ipynb) + `outputs/`
  (`vectorizer.joblib`, `duplicate_map.csv`, `model_selection.json`, optionally
  `embeddings_all.npz`). It picks the best candidate by precision@5 and builds the matching
  ranker (`tfidf` / `embeddings`). Falls back to TF-IDF if `model_selection.json`
  is absent.
- User query → rank → top-k cards (title link, tags, explicit % match, body preview).
- `requirements.txt` includes `sentence-transformers` (pulls torch — heavy; if the winner is
  TF-IDF, deploy is much lighter, which is a legitimate deployment argument in the report).
- Host on Streamlit Community Cloud or Hugging Face Spaces. **What gets committed/deployed**
  (per `.gitignore`): all code + notebooks + `requirements.txt` + `dataset/questions_clean.csv`
  + `outputs/{vectorizer.joblib, duplicate_map.csv, model_selection.json, embeddings_all.npz}`.
  Everything else (`dataset/questions.csv` 1.1 GB, `outputs/X_*.npz` ~500 MB, logs, configs)
  stays local. Provide the public link in the report (5 marks).
- The clean CSV is ~143 MB, dominated by `body_text`; if the host has a repo size limit,
  truncate `body_text` to a preview column in the Phase 9 save cell (the app already shows
  only the first 500 chars).

---

## 6. QA checklist (after every run)

1. Row counts at each phase boundary match the log.
2. Zero unexpected nulls; dtypes correct.
3. 20-row random sample eyeballed after cleaning.
4. No raw entities/URLs in `text_all` (placeholders only).
5. Label matrix shape == vocab size; every row has ≥ 1 label or `<OTHER>`.
6. Split strictly temporal; no ID leakage.
7. Vectorizer fitted on train only; artifacts reload in a fresh process; Streamlit app runs
   from the persisted artifacts alone.

---

## 7. Open decisions (defaults proposed)

| Decision | Default | Config key |
|---|---|---|
| Collection scope | One pass 2024–2026, all tags; AI/ML filter in Phase 1 (first) | `START_DATE`/`END_DATE`, `domain_tags` |
| Drop duplicate-closed from candidates? | Yes, keep redirect map | `keep_duplicate_redirect` |
| Drop off-topic closed? | Yes | `drop_offtopic_closed` |
| Minimum tag frequency (labels) | 5 | `min_tag_freq` |
| Minimum body length / views | 50 chars / 5 views | `min_body_chars`, `min_view_count` |
| Serve only answered questions as recommendations? | Yes (rank by `has_accepted_answer`) | `catalog_answered_only` |
| English-only? | Yes | `english_only` |
| Split dates | temporal, see Phase 10 | `split_dates` |
| Deployed model | Winner of the test-split comparison, recorded in `model_selection.json` and auto-loaded by `app.py` (TF-IDF fallback) | `model_selection.json` |

---

## 8. Risks & notes

- **Angle brackets are not HTML** (1,131 rows) — clean conservatively or code-in-prose is destroyed.
- **Long-tail tags** — mitigated by tag-scoped collection; `min_tag_freq` + `<OTHER>` as safety net.
- **Quota limits**: collection is quota-bound (10,000 requests/day with key, 100 questions each);
  the resumable collector logs `quota_remaining` and restarts from the newest `creation_date` in
  the CSV, so a quota/error stop never re-crawls old days. The full 2024–2026 crawl (528,495
  questions) is complete.
- **Cross-tag duplicates** are prevented at write time by the seen-ID set; Phase 1 still asserts
  `question_id` uniqueness as a guard.
- **12.5% of AI/ML questions have no code** (no code ≠ invalid — kept with a `has_code` flag; char n-grams + the flag keep text-only items competitive).
- **Duplicate redirect targets mostly absent in-dataset** (19/1,040) — nice-to-have, not core.
- **Small recent months** (Nov 2026 partial) — with multi-year data the temporal test split will
  be large enough; set split dates after final collection.
- Deterministic pipeline required: the report must be reproducible by another student.
