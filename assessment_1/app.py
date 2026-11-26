"""StackLearn — Stack Overflow AI/ML similar-question recommender.

Loads the clean catalog (dataset/questions_clean.csv, written by data_preprocessing.ipynb) and the
model artifacts (outputs/, written by train_model.ipynb); serves the top-k most similar
already-answered questions for a question title.

Run locally:  streamlit run app.py
Deploy:       Streamlit Community Cloud / Hugging Face Spaces (see requirements.txt)
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="StackLearn", layout="wide")  # must be the first st command

OUT = "outputs"
CLEAN = "dataset/questions_clean.csv"
TOP_K = 10


@st.cache_resource
def load_artifacts():
    """Load the cleaned catalog and the best model chosen by train_model.ipynb.

    train_model.ipynb writes model_selection.json (TF-IDF vs sentence embeddings, scored on
    the test split). We deploy whichever won; if the file is absent we fall back to TF-IDF.
    """
    df = pd.read_csv(CLEAN)
    df["question_id"] = df["question_id"].astype(str)
    df["creation_date"] = pd.to_datetime(df["creation_date"], utc=True)
    df["tag_list"] = df["tags"].str.split("|")
    vectorizer = joblib.load(f"{OUT}/vectorizer.joblib")
    dup_map = pd.read_csv(f"{OUT}/duplicate_map.csv")
    dup_map = dict(zip(dup_map["question_id"].astype(str), dup_map["duplicate_of"]))

    model_name, emb_model, E, X = "tfidf", None, None, None
    sel_path = f"{OUT}/model_selection.json"
    if os.path.exists(sel_path):
        sel = json.load(open(sel_path))
        model_name = max(sel["model_results"], key=lambda n: sel["model_results"][n]["5"][0])
        if model_name not in ("tfidf", "embeddings"):  # e.g. older json with a removed candidate
            model_name = "tfidf"
        if model_name == "embeddings":
            from sentence_transformers import SentenceTransformer

            emb_model = SentenceTransformer(sel["emb_model"])
            E = np.load(f"{OUT}/embeddings_all.npz")["E"]  # precomputed catalog embeddings
    if model_name == "tfidf":
        X = vectorizer.transform(df["text_all"])  # catalog vectors, same cleaning as training
    return df, vectorizer, X, dup_map, model_name, emb_model, E


def make_ranker(model_name, vectorizer, X, emb_model, E):
    """Return rank(query) -> similarity of every catalog question to the query."""
    if model_name == "embeddings":
        return lambda q: cosine_similarity(emb_model.encode([q]), E)[0]
    return lambda q: cosine_similarity(vectorizer.transform([q]), X)[0]


df, vectorizer, X, dup_map, model_name, emb_model, E = load_artifacts()
rank = make_ranker(model_name, vectorizer, X, emb_model, E)

st.title("StackLearn — similar AI/ML Stack Overflow questions")
st.caption(
    f"Catalog: {len(df):,} answered AI/ML questions from Stack Overflow "
    f"({df['creation_date'].min():%Y-%m-%d} to {df['creation_date'].max():%Y-%m-%d}). "
    f"Model: {model_name}. "
    "Type a question title and get the most similar solved questions."
)

# Live search: the on_change callback re-runs the app on every edit and on Enter.
def _on_search_change():
    st.session_state["query"] = st.session_state["search_box"]


st.text_input(
    "Search",  # required, but hidden — the placeholder carries the meaning
    placeholder="Search questions here...",
    label_visibility="hidden",
    key="search_box",
    on_change=_on_search_change,
)

raw = st.session_state.get("query", "")
query = (raw + " " + raw).lower().strip()  # title counted twice, same as in training
if not query:
    st.caption("Type a question title — results update as you type.")
else:
        sims = rank(query)
        order = np.argsort(-sims)[:TOP_K]
        results = df.iloc[order].copy()
        results["similarity"] = sims[order]

        st.subheader(f"Top {len(results)} similar questions")
        for _, r in results.iterrows():
            link = f"https://stackoverflow.com/questions/{r['question_id']}"
            accepted = " | accepted answer" if r["has_accepted_answer"] else ""
            dup_note = ""
            if r["question_id"] in dup_map:
                targets = dup_map[r["question_id"]].split("|")
                dup_note = f" | duplicate of {', '.join(targets)}"
            st.markdown(
                f"**{r['similarity'] * 100:.1f}% match** — [{r['title']}]({link})  \n"
                f"tags: `{r['tags']}`{accepted}{dup_note}"
            )
            st.write(str(r["body_text"])[:500])  # body shown directly, no click needed
