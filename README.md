# Applied Artificial Intelligence — Group Assessment

Coursework repository for **CS5079 Applied Artificial Intelligence** (University of Aberdeen, MSc AI).
Contains two assessments, each in its own folder, with the corresponding brief in the matching PDF at the repo root.

```
assessment_group/
├── assessment_1/          # CA1 — StackLearn: Stack Overflow similar-question recommender
│   ├── config.json                # single source of truth: tags, dates, thresholds, paths, top-k
│   ├── data_collection.ipynb       # crawl Stack Overflow → dataset/questions.csv (raw)
│   ├── data_preprocessing.ipynb    # clean/filter raw data → dataset/questions_clean.csv
│   ├── train_model.ipynb           # train & compare retrievers → outputs/ artifacts
│   ├── app.py                      # Streamlit app (StackLearn)
│   ├── preprocessing_plan.md       # design document for the pipeline
│   ├── requirements.txt
│   ├── dataset/                    # raw + clean question catalogs (CSV)
│   └── outputs/                    # model artifacts loaded by app.py
├── assessment_2/          # CA2 — dataset for the second assessment (in progress)
├── assessment_1_ques.pdf  # CA1 brief
└── assessment_2_ques.pdf  # CA2 brief
```

## Assessment 1 — StackLearn

**StackLearn** is a semantic duplicate-question retrieval system for AI/ML developers: given a new
question title, it retrieves the **top 10 most similar already-solved** Stack Overflow questions
(title, tags, score, answer count, link), so a developer can reuse an existing solution instead of
posting a duplicate and waiting for a new answer. It is a content-based retrieval system over a
catalog of answered AI/ML questions — no user accounts, no feedback loop.

### Pipeline

All shared constants — collection dates, domain tags, preprocessing thresholds, split dates,
TF-IDF hyperparameters, top-k values, serving top-k — live in **`config.json`** (single source of
truth). Every notebook and `app.py` loads it; nothing writes it.

Each notebook has one job and feeds the next:

1. **`data_collection.ipynb`** — crawls Stack Overflow questions (2024–2026) via the Stack Exchange API
   into `dataset/questions.csv` (~528k rows raw).
2. **`data_preprocessing.ipynb`** — filters to 26 curated AI/ML tags, keeps only answered questions,
   cleans text, deduplicates, and engineers features → `dataset/questions_clean.csv`
   (36,060 rows) plus runtime artifacts in `outputs/`.
3. **`train_model.ipynb`** — builds and compares three retrievers on a leakage-free split:
   TF-IDF, sentence-transformers embeddings (`all-MiniLM-L6-v2`), and a hybrid;
   writes `outputs/model_selection.json`. The best model won at ~0.99 Recall@5 on the test split.
4. **`app.py`** — loads the clean catalog and the winning model, serves the top 10 similar
   answered questions for a user-supplied question title.

### Run locally

Requires Python 3.10+.

```bash
cd assessment_1
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
streamlit run app.py
```

The app needs `dataset/questions_clean.csv` and the runtime artifacts in `outputs/`
(`vectorizer.joblib`, `model_selection.json`, `embeddings_all.npz`).
If any are missing, regenerate them by running the notebooks in order
(`data_collection.ipynb` → `data_preprocessing.ipynb` → `train_model.ipynb`).

> Note: the raw crawl (`dataset/questions.csv`, ~1.1 GB) and large training matrices
> (e.g. `X_train.npz`, ~500 MB) are regenerable and not tracked in git — see
> `assessment_1/.gitignore`.

## Assessment 2

Contains `SAdata_allMeasures.csv`, the dataset for the second assessment brief
(`assessment_2_ques.pdf`). Work in progress.
