"""Generate every figure used in report.tex from the pipeline's own artifacts.

Run from assessment_1/:   python report/make_figures.py
Inputs : config.json, dataset/questions_clean.csv, outputs/{model_selection.json,
         embeddings_all.npz, vectorizer.joblib}  (written by the three notebooks)
Outputs: report/figures/*.png

"""
import json
import os
from collections import Counter

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

FIG = "report/figures"
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, GREY, INK, INK2 = "#257de0", "#f47230", "#8a8a8a", "#191919", "#666666"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.bbox": "tight", "savefig.dpi": 200,
})

config = json.load(open("config.json"))
domain_tags = set(config["domain_tags"])
df = pd.read_csv(config["clean_path"])
df["creation_date"] = pd.to_datetime(df["creation_date"], utc=True)
df["domain_tag_list"] = df["tags"].str.split("|").map(lambda t: sorted(set(t) & domain_tags))
selection = json.load(open(f"{config['output_dir']}/model_selection.json"))


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", f"{FIG}/{name}.png")


# 1. Preprocessing funnel (counts printed by data_preprocessing.ipynb)
stages = [("Raw crawl", 528495), ("AI/ML tag filter", 23682), ("Not closed", 22725),
          ("English only", 22719), ("Body $\\geq$ 50 chars", 22686), ("Has $\\geq$ 1 answer", 16137)]
fig, ax = plt.subplots(figsize=(6, 2.2))
labels, counts = zip(*stages)
ax.barh(range(len(stages))[::-1], counts, color=BLUE, height=0.6)
ax.set_xscale("log")
ax.set_yticks(range(len(stages))[::-1], labels)
ax.set_xlabel("Questions (log scale)")
ax.grid(axis="y", visible=False)
for y, c in zip(range(len(stages))[::-1], counts):
    ax.text(c * 1.08, y, f"{c:,}", va="center", color=INK, fontsize=8)
ax.set_xlim(1e4, 2e6)
save(fig, "funnel")

# 2. Top domain tags in the clean catalog
top = Counter(t for ts in df["domain_tag_list"] for t in ts).most_common(15)
fig, ax = plt.subplots(figsize=(6, 2.6))
ax.barh([t for t, _ in top][::-1], [c for _, c in top][::-1], color=BLUE, height=0.6)
ax.set_xlabel("Questions carrying the tag")
ax.grid(axis="y", visible=False)
save(fig, "tags")

# 3. Source trend, from the RAW crawl (dataset/questions.csv, before any filtering):
#    (a) monthly volume indexed to Jan 2024 = 100, all questions vs AI/ML-tagged questions
#        (an index keeps both on one comparable axis), (b) share of AI/ML questions answered.
import matplotlib.dates as mdates

raw = pd.read_csv(config["src_path"], usecols=["creation_date", "tags", "answer_count"], low_memory=False)
raw["creation_date"] = pd.to_datetime(raw["creation_date"], utc=True, format="ISO8601")
raw = raw[(raw["creation_date"] >= "2024-01-01") & (raw["creation_date"] < "2026-11-01")]  # whole months only
raw["is_ai"] = raw["tags"].str.split("|").map(lambda t: bool(set(t) & domain_tags))
raw = raw.set_index("creation_date")
all_monthly = raw.resample("MS").size()
ai_monthly = raw[raw["is_ai"]].resample("MS").size()
answered_rate = raw[raw["is_ai"]]["answer_count"].gt(0).resample("MS").mean() * 100
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.4, 2.2))
ax_a.plot(all_monthly.index, all_monthly / all_monthly.iloc[0] * 100, color=GREY, linewidth=2, label="All questions")
ax_a.plot(ai_monthly.index, ai_monthly / ai_monthly.iloc[0] * 100, color=BLUE, linewidth=2, label="AI/ML questions")
ax_a.set_ylabel("Volume (Jan 2024 = 100)")
ax_a.set_ylim(0, 110)
ax_a.legend(frameon=False, loc="upper right")
ax_a.set_title("(a) Questions asked per month", fontsize=9)
ax_b.plot(answered_rate.index, answered_rate, color=BLUE, linewidth=2)
ax_b.set_ylabel("Answered (%)")
ax_b.set_ylim(0, 100)
ax_b.set_title("(b) AI/ML questions answered", fontsize=9)
for ax in (ax_a, ax_b):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
save(fig, "trend")
print(f"trend: all {all_monthly.iloc[0]} -> {all_monthly.iloc[-1]}, AI/ML {ai_monthly.iloc[0]} -> {ai_monthly.iloc[-1]}, "
      f"AI share {ai_monthly.sum() / all_monthly.sum():.3f}, answered {answered_rate.iloc[0]:.0f}% -> "
      f"{answered_rate.iloc[-1]:.0f}% (mean to Jun 2026 {answered_rate[:'2026-06'].mean():.0f}%)")

# 4. Model comparison on the test split (from model_selection.json)
metrics = [("P@5", "5", 0), ("P@10", "10", 0), ("Hit@5", "5", 1), ("MRR@10", "10", 2)]
res = selection["model_results"]
x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(6, 2.4))
for off, (name, label, color) in zip((-0.18, 0.18), [("tfidf", "TF-IDF", ORANGE),
                                                     ("embeddings", "MiniLM embeddings", BLUE)]):
    vals = [res[name][k][i] for _, k, i in metrics]
    ax.bar(x + off, vals, width=0.34, color=color, label=label)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
ax.axhline(res["tfidf"]["5"][4], color=GREY, linestyle="--", linewidth=1)
ax.text(3.42, res["tfidf"]["5"][4] + 0.015, "random\nP@k", color=INK2, fontsize=7.5, ha="left", va="bottom")
ax.set_xlim(-0.5, 3.95)
ax.set_xticks(x, [m for m, _, _ in metrics])
ax.set_ylim(0, 1.0)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "models")

# 5. Catalog-size experiment: P@5 when the catalog is a random fraction of the train split
def precision_at_k(sim, query_tags, catalog_tags, k=5):
    top_k = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([np.mean([bool(catalog_tags[j] & q) for j in row])
                          for row, q in zip(top_k, query_tags)]))

idx = np.arange(len(df))
train_idx, rest_idx = train_test_split(idx, test_size=0.2, random_state=24)
_, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=24)
test_idx = test_idx[:500]                                  # MAX_EVAL_QUERIES in the notebook
tag_sets = df["domain_tag_list"].map(set).tolist()
E = np.load(f"{config['output_dir']}/embeddings_all.npz")["E"]
vectorizer = joblib.load(f"{config['output_dir']}/vectorizer.joblib")
X = vectorizer.transform(df["text_all"])
fractions = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
rng = np.random.default_rng(0)
curves = {"tfidf": [], "embeddings": []}
for f in fractions:
    sub = rng.permutation(train_idx)[: int(len(train_idx) * f)]
    cat_tags, q_tags = [tag_sets[i] for i in sub], [tag_sets[i] for i in test_idx]
    curves["embeddings"].append(precision_at_k(cosine_similarity(E[test_idx], E[sub]), q_tags, cat_tags))
    curves["tfidf"].append(precision_at_k(cosine_similarity(X[test_idx], X[sub]), q_tags, cat_tags))
    print(f"catalog {len(sub):>6}: tfidf={curves['tfidf'][-1]:.3f} emb={curves['embeddings'][-1]:.3f}")
sizes = [int(len(train_idx) * f) for f in fractions]
fig, ax = plt.subplots(figsize=(6, 2.2))
ax.plot(sizes, curves["embeddings"], color=BLUE, linewidth=2, marker="o", markersize=4, label="MiniLM embeddings")
ax.plot(sizes, curves["tfidf"], color=ORANGE, linewidth=2, marker="o", markersize=4, label="TF-IDF")
ax.set_xlabel("Catalog size (questions searched)")
ax.set_ylabel("Test P@5")
ax.legend(frameon=False, loc="lower right")
save(fig, "catalog_size")
json.dump({"sizes": sizes, **curves}, open(f"{FIG}/catalog_size.json", "w"), indent=1)

# 6. Realistic queries: users type a short title, but the notebook queries with title + body.
#    Re-score both models with title-only queries (full train catalog), overall and per frequent tag.
from sentence_transformers import SentenceTransformer

def per_query_p5(sim):
    return np.array([np.mean([bool(tag_sets[train_idx[j]] & tag_sets[q]) for j in row])
                     for row, q in zip(np.argsort(-sim, axis=1)[:, :5], test_idx)])

titles = df["title"].iloc[test_idx].str.lower().tolist()
encoder = SentenceTransformer(selection["emb_model"])
p5 = {"embeddings": per_query_p5(cosine_similarity(encoder.encode(titles), E[train_idx])),
      "tfidf": per_query_p5(cosine_similarity(vectorizer.transform(titles), X[train_idx]))}
diff = p5["embeddings"] - p5["tfidf"]
title_only = {name: round(float(v.mean()), 3) for name, v in p5.items()}
title_only["gap"] = round(float(diff.mean()), 3)
title_only["paired_z"] = round(float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))), 1)
per_tag = {}
for tag in sorted({t for q in test_idx for t in tag_sets[q]}):
    rows = [n for n, q in enumerate(test_idx) if tag in tag_sets[q]]
    if len(rows) >= 25:                                            # enough queries for a stable mean
        per_tag[tag] = {"n": len(rows), **{name: round(float(v[rows].mean()), 3) for name, v in p5.items()}}
json.dump({"title_only_p5": title_only, "per_tag_title_p5": per_tag},
          open(f"{FIG}/extra_results.json", "w"), indent=1)
print("title-only P@5:", title_only)
for tag, r in sorted(per_tag.items(), key=lambda x: -x[1]["n"]):
    print(f"  {tag:22s} n={r['n']:3d} emb={r['embeddings']:.3f} tfidf={r['tfidf']:.3f}")
