# Project Plan: Stack Overflow Question Recommendation & Tag Suggestion System (Task 1)

## 1. Assessment Requirements & Course Alignment

Based on `CS5079_Assessment_1_2026-2027.pdf`, **Task 1 (30 Marks, 1,500 words max)** requires building, evaluating, and deploying an end-to-end AI product on top of our collected dataset (`questions_2026.csv.zip`, **20,142 Stack Overflow questions** collected via `data_collection.ipynb`).

To achieve **CGS A** (*"Key AI techniques delivered in the course were used in this project... All of the components in those guides described in the Task were working perfectly, with perfect justification"*), our separate Python notebook (`question_recommender_system.ipynb`) directly incorporates the methods taught in the `resources/` lectures and tutorials, and includes an explicit **"Analysis of Results"** block after every single experiment and figure.

### Mapping Course Resources (`resources/`) to Our Implementation

| Course Resource | Techniques Taught | How It Is Used in `question_recommender_system.ipynb` |
| :--- | :--- | :--- |
| **`Lecture2-Working with Features.pdf`** & **`Tutorial2-Solutions.ipynb`** | Numeric log transforms (`np.log1p`), `Binarizer`, `PolynomialFeatures`, quantile/fixed-width binning (`pd.cut`, `pd.qcut`), temporal feature evaluation, feature selection (dropping irrelevant calendar attributes as in *Tutorial 2 Task 2.2*), `StandardScaler` / `MinMaxScaler` / `RobustScaler`, PCA dimensionality reduction, correlation heatmaps. | **EDA, Feature Engineering & Selection (Sections 2 & 3)**: Empirically evaluate and drop uninformative clock/calendar features (`Month`, `Day`, `DayOfWeek`, `Hour`, `TimeOfDayBin`) while retaining recency (`question_age_days`, `views_per_day_log`); apply `np.log1p`, `RobustScaler`, and `MinMaxScaler` to build an age-normalized **Question Quality Score (`quality_score`)**; use `PCA` to visualize embeddings. |
| **`Tutorial3-solutions.ipynb`** | IQR outlier analysis ($Q_1 - 1.5 \cdot IQR$, $Q_3 + 1.5 \cdot IQR$), recency (`reference_day - creation_date`), `KMeans` clustering, `silhouette_score` & `silhouette_samples` analysis, cluster profiling. | **Outlier Analysis (Section 2.3), Recency (Section 3.1) & Topic Clustering (Section 4.4)**: Quantify IQR outliers (explaining why viral questions are log-transformed/robust-scaled rather than deleted), compute question age/recency, and cluster questions using `KMeans` + `silhouette_score`. |
| **`Tutorial4-solution.ipynb`** | Grouped descriptive statistics (`describe`), One-Way ANOVA F-test (`scipy.stats.f_oneway`), boxplots, `GridSearchCV`, `classification_report`, `confusion_matrix`, feature importances / model interpretation. | **Statistical Testing (Section 2.5) & Supervised Evaluation/Tuning (Sections 4 & 5)**: Run `stats.f_oneway` across resolution categories (`Accepted Answer`, `Answered (Unaccepted)`, `Unanswered`), tune SVM hyperparameters with `GridSearchCV(cv=5)`, and explain model predictions. |
| **`lecture5_SentimentPrediction.pdf`** & **`Lec_16_SVM(11).pdf`** | NLP preprocessing (`unicodedata` NFKD accented char removal, `contractions`, regex cleaning, `ToktokTokenizer`, `nltk` stopwords retaining `'no'`/`'not'`, lemmatization), **Bag of Words (`CountVectorizer`)**, **TF-IDF (`TfidfVectorizer` with `sublinear_tf=True`, `ngram_range=(1,2)`)**, **Word2Vec** + `averaged_word2vec_vectorizer`, **Cosine Similarity**, **Linear Support Vector Machine (`LinearSVC`)**, and **Sequential Deep Neural Network (DNN)**. | **Core NLP Normalization (Section 3.4), Candidate Recommendation Models & Tag Prediction (Sections 4 & 5)**: Clean `title` and `body_text` while preserving technical tokens (`c++`, `c#`, `.net`, `node.js`); compare **BoW**, **TF-IDF**, **Averaged Word2Vec**, and **SentenceTransformer** retrieval models alongside **SVM vs. Deep Neural Network** supervised tag classifiers. |
| **`lecture5_SentimentPrediction_LLM.pdf`** | Pretrained representation/embedding models using `sentence-transformers` (`SentenceTransformer`), dense semantic embeddings, two-step embedding + lightweight classifier. | **Semantic Question Recommender (Sections 4 & 5)**: Encode questions into dense $384$-D semantic vectors using `SentenceTransformer('all-MiniLM-L6-v2')` so queries match semantically similar questions even when exact keywords differ. |
| **`Lecture4-Hyperparameters_ModelInterpret_Deployment.pdf`** | `GridSearchCV`, Bias-Variance Trade-off, model interpretability (Global & Local explanations), **Model Persistence (`joblib.dump`)**, and **Streamlit Deployment (`app.py`, `prediction.py`, `requirements.txt`)**. | **Evaluation, Explainability & Deployment (Sections 5 & 6)**: Tune models, interpret recommendations (shared TF-IDF terms + semantic similarity + quality breakdown + SVM hyperplane weights), export artifacts via `joblib.dump`, and deploy via Streamlit. |

---

## 2. Notebook Structure & Implementation Status (All Sections 1 to 6 Completed)

### ✅ Section 1: Project Definition & Dataset Sufficiency Justification (Task 1.1 — 5 Marks)
- **1.1 Product Definition & Usefulness**: Defines the end-to-end **Stack Overflow Question Recommendation & Tag Suggestion System** (semantic + lexical question retrieval, automated technical tag suggestion, and quality-aware re-ranking).
- **1.2 Dataset Loading & Sufficiency Table**: Loads `questions_2026.csv.zip` (**20,142 questions**, 13 columns, `10,027` unique tags, `73.2%` with code blocks, `46.1%` answered, `25.4%` accepted, `808` duplicate links).
- **Analysis of Section 1 Results**: In-depth justification of sample volume, multi-modal text/code richness, and ground-truth labels.

### ✅ Section 2: Exploratory Data Analysis (EDA) & Data Cleaning (Task 1.2 & 1.3)
- **2.1 & 2.2 Missing Values, Imputation & Type Casting**: Evaluates missingness across all 13 attributes, distinguishes structural missingness (`closed_reason` `89.27%`, `duplicate_of` `95.99%`, `code_text` `26.77%`) from accidental missingness (`body_text` `0.02%`), casts `creation_date` to UTC timestamps, and derives `resolution_status` (`Accepted Answer` `25.43%`, `Answered (Unaccepted)` `48.88%`, `Unanswered` `25.69%`).
- **2.3 Descriptive Statistics & IQR Outlier Analysis**: Computes summary statistics and $1.5 \times IQR$ bounds (`score` `19.90%` outliers, `view_count` `7.13%` outliers), critically justifying why viral Stack Overflow questions must be retained and log-transformed/robust-scaled rather than deleted.
- **2.4 Tag Distribution & Meta-Tag Discovery**: Visualizes Top 20 tags and resolution status, identifying cross-cutting meta-tags (`advice`, `best-practices`, `tooling-recommendation`) that must be separated from concrete programming language tags.
- **2.5 One-Way ANOVA F-Tests (`scipy.stats.f_oneway`) & Boxplots**: Proves statistically significant differences ($p < 0.01$) across resolution categories for `score` ($F = 366.97$), `has_code` ($F = 342.19$), `view_count` ($F = 99.83$), `body_word_count` ($F = 51.85$), and `title_word_count` ($F = 4.88$).
- **2.6 Correlation Matrix Heatmap**: Demonstrates that log-transforming views (`view_count_log`) nearly doubles correlation with `answer_count` ($r = 0.22 \rightarrow 0.40$) and quadruples correlation with `is_answered` ($r = 0.08 \rightarrow 0.29$), with no severe multicollinearity ($r < 0.65$).

### ✅ Section 3: Data Preprocessing, Feature Engineering & Selection (Task 1.3 — 3 Marks)
- **3.1 Temporal Feature Evaluation, Selection & Numeric Engineering**:
  - Extracts calendar/clock features (`Month`, `Day`, `DayOfWeek`, `Hour`, `TimeOfDayBin`) alongside **Question Recency (`question_age_days`)** and **Age-Normalized View Rate (`views_per_day`, `views_per_day_log`)**.
  - Empirically proves that `Hour` ($r = -0.0014$) and `DayOfWeek` ($r = 0.0067$) have near-zero correlation with engagement and no semantic relevance to user queries, explicitly dropping `['Month', 'Day', 'DayOfWeek', 'Hour', 'TimeOfDayBin']` while retaining `question_age_days` and `views_per_day_log`.
  - Applies `Binarizer`, `np.log1p`, `PolynomialFeatures` interaction terms, fixed-width length binning (`pd.cut`), and adaptive quartile binning (`pd.qcut`).
- **3.2 Feature Scaling & Composite Question Quality Score**:
  - Compares `StandardScaler`, `MinMaxScaler`, and `RobustScaler`, and constructs a bounded $[0, 1]$ age-adjusted **`quality_score`**.
- **3.3 Categorical Encoding & Primary Domain Tag Extraction**:
  - Extracts the Top 10 technical programming domains (`python`, `c++`, `java`, `javascript`, `c#`, `android`, `c`, `r`, `html`, `ios` — covering `46.61%` of all questions) and applies `LabelEncoder` and `OneHotEncoder`.
- **3.4 NLP Text Normalization Pipeline**:
  - Applies `remove_accented_chars` (`NFKD`), `expand_text_contractions`, custom `remove_special_characters_technical` (preserving `c++`, `c#`, `.net`, `node.js`), `ToktokTokenizer` stopword removal (retaining `'no'` and `'not'`), and noun+verb `WordNetLemmatizer`.
  - Quantifies vocabulary compression (**`-44.9%` total tokens**, **`-52.8%` unique vocabulary size**) and saves `dataset/questions_2026_processed.pkl`.

### ✅ Section 4: Candidate Machine Learning & AI Models (Task 1.3 — 3 Marks)
- **Part A — 5 Candidate Question Retrieval & Recommendation Models**:
  1. **Model 1: Bag-of-Words (`CountVectorizer`, unigrams+bigrams) + Cosine Similarity** (*Lecture 5 Slide 19*)
  2. **Model 2: Sublinear TF-IDF (`TfidfVectorizer`, `sublinear_tf=True, ngram_range=(1,2)`) + Cosine Similarity** (*Lecture 5 Slide 20*)
  3. **Model 3: Spectral / LSA Dense Word Embeddings (`TruncatedSVD`, 200d) + `averaged_word2vec_vectorizer` + Cosine Similarity** (*Lecture 5 Slides 26–28*)
  4. **Model 4: Pretrained Transformer Embeddings (`SentenceTransformer('all-MiniLM-L6-v2')`, 384d) + Cosine Similarity** (*Lecture 5 LLM Slides 20–23*)
  5. **Model 5: Hybrid Quality-Aware Recommender (`TF-IDF + SentenceTransformer + quality_score`) + Unsupervised `KMeans` Topic Clustering** (*Tutorial 3 & Lecture 2 PCA Slides 50–55*)
- **Part B — 2 Candidate Supervised Technical Tag Classification Models (SVM vs. Deep Neural Network)**:
  1. **Candidate Classifier 1: Support Vector Machine (`LinearSVC`)** (*SVM Lecture & Lecture 5 Slide 22*)
  2. **Candidate Classifier 2: Deep Neural Network (3-Layer Sequential MLP with `BatchNorm`, `ReLU`, `Dropout(0.35)`)** (*Lecture 1 Slide 42 & Lecture 5 Slides 23–24*)

### ✅ Section 5: Model Training, Hyperparameter Tuning, Evaluation & Selection for Deployment (Task 1.3 — 9 Marks)
- **5.1 Leakage-Free Quantitative Benchmark of the 5 Candidate Recommendation Models ($1,000$ Test Queries)**:
  - **Model 1 (BoW)**: `51.0%` Hit@1, `65.7%` Hit@5, `71.2%` Hit@10, `0.5754` MRR@10, `41.48%` Tag Precision@5, `27.40%` Accepted Ans@5.
  - **Model 2 (Sublinear TF-IDF)**: `55.1%` Hit@1, `75.3%` Hit@5, `80.4%` Hit@10, `0.6349` MRR@10, `55.18%` Tag Precision@5, `28.68%` Accepted Ans@5.
  - **Model 3 (Spectral/LSA Embeddings 200d)**: `51.7%` Hit@1, `70.0%` Hit@5, `76.2%` Hit@10, `0.5925` MRR@10, `46.12%` Tag Precision@5, `30.38%` Accepted Ans@5.
  - **Model 4 (SentenceTransformer MiniLM-L6-v2, Body-Only)**: `78.1%` Hit@1, `90.0%` Hit@5, `92.6%` Hit@10, `0.8322` MRR@10, `64.84%` Tag Precision@5, `29.54%` Accepted Ans@5.
  - **Model 5 (Hybrid: TF-IDF + Transformer + Quality)**: **`80.9%` Hit@1 (Best)**, **`92.4%` Hit@5 (Best)**, **`94.4%` Hit@10 (Best)**, **`0.8595` MRR@10 (Best)**, **`67.72%` Tag Precision@5 (Best)**, **`0.3911` Avg Quality@5 (Best)**, and **`44.54%` Accepted Ans@5 (`+15.0%` gain over pure Transformer)** at **`3.54 ms/query`**.
- **5.2 Leakage-Free Supervised Tag Classification Comparison (SVM vs. Deep Neural Network)**:
  - **Tuned Support Vector Machine (`LinearSVC`, interior optimum $C=0.5$ via 5-Fold `GridSearchCV`)**: **`84.82%` Test Accuracy**, **`83.66%` Macro Precision**, **`83.06%` Macro-F1**, **`84.73%` Weighted F1**, **`0.0045 ms/sample` inference latency**.
  - **3-Layer Sequential Deep Neural Network (PyTorch MLP with Early Stopping)**: **`83.76%` Test Accuracy**, **`81.31%` Macro Precision**, **`81.69%` Macro-F1**, **`83.71%` Weighted F1**, **`0.1865 ms/sample` inference latency**.
- **5.3 & 5.4 Model Interpretation & Deployment Selection Justification**:
  - Global glassbox interpretation of SVM hyperplane weights (`coef_`) across all 10 programming classes, plus a comprehensive multi-criteria justification table for why **Model 5 (Hybrid Recommender) + Calibrated SVM Tag Classifier** is selected for production deployment.

### ✅ Section 6: Model Persistence & Streamlit Deployment (Task 1.4 — 5 Marks)
- **6.1 Model Persistence (`joblib.dump`)**: All deployment artifacts exported to `models/` (`41.71 MB` total: `tfidf_vectorizer.joblib`, `tfidf_matrix.joblib`, `tag_classifier_svm.joblib`, `topic_kmeans.joblib`, `questions_metadata.joblib`, `st_embeddings_fp16.npy`).
- **6.2 End-to-End Inference & Local Explainability Showcase**: Demonstrated in Cell 31 on realistic Python/Pandas and C++20 smart-pointer developer queries.
- **6.3 Streamlit Deployment Package (`prediction.py`, `app.py`, `requirements.txt`)**: Created in the workspace root following *CS5079 Lecture 4 Slides 29–32*.
