# Project Plan: Stack Overflow Question Recommendation & Tag Suggestion System (Task 1)

## 1. Assessment Requirements & Course Alignment

Based on `CS5079_Assessment_1_2026-2027.pdf`, **Task 1 (30 Marks, 1,500 words max)** requires building, evaluating, and deploying an end-to-end AI product on top of our collected dataset (`questions_2026.csv.zip`, **20,142 Stack Overflow questions** collected via `data_collection.ipynb`).

To achieve **CGS A** (*"Key AI techniques delivered in the course were used in this project... All of the components in those guides described in the Task were working perfectly, with perfect justification"*), our separate Python notebook (`question_recommender_system.ipynb`) directly incorporates the methods taught in the `resources/` lectures and tutorials, and includes an explicit **"Analysis of Results"** block after every single experiment and figure.

### Mapping Course Resources (`resources/`) to Our Implementation

| Course Resource | Techniques Taught | How It Is Used in `question_recommender_system.ipynb` |
| :--- | :--- | :--- |
| **`Lecture2-Working with Features.pdf`** & **`Tutorial2-Solutions.ipynb`** | Numeric log transforms (`np.log1p`), `Binarizer`, `PolynomialFeatures`, quantile/fixed-width binning (`pd.cut`, `pd.qcut`), temporal feature evaluation, feature selection (dropping irrelevant calendar attributes as in *Tutorial 2 Task 2.2*), `StandardScaler` / `MinMaxScaler` / `RobustScaler`, PCA dimensionality reduction, correlation heatmaps. | **EDA, Feature Engineering & Selection (Sections 2 & 3)**: Empirically evaluate and drop uninformative clock/calendar features (`Month`, `Day`, `DayOfWeek`, `Hour`, `TimeOfDayBin`) while retaining recency (`question_age_days`, `views_per_day_log`); apply `np.log1p`, `RobustScaler`, and `MinMaxScaler` to build an age-normalized **Question Quality Score (`quality_score`)**; use `PCA` to visualize embeddings. |
| **`Tutorial3-solutions.ipynb`** | IQR outlier analysis ($Q_1 - 1.5 \cdot IQR$, $Q_3 + 1.5 \cdot IQR$), recency (`reference_day - creation_date`), `KMeans` clustering, `silhouette_score` & `silhouette_samples` analysis, 3D cluster visualization. | **Outlier Analysis (Section 2.3), Recency (Section 3.1) & Topic Clustering (Section 4)**: Quantify IQR outliers (explaining why viral questions are log-transformed/robust-scaled rather than deleted), compute question age/recency, and cluster questions using `KMeans` + `silhouette_score`. |
| **`Tutorial4-solution.ipynb`** | Grouped descriptive statistics (`describe`), One-Way ANOVA F-test (`scipy.stats.f_oneway`), boxplots, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GridSearchCV`, `classification_report`, `confusion_matrix`, feature importances / model interpretation. | **Statistical Testing (Section 2.5) & Supervised Evaluation/Tuning (Sections 4 & 5)**: Run `stats.f_oneway` across resolution categories (`Accepted Answer`, `Answered (Unaccepted)`, `Unanswered`), compare supervised tag classifiers, tune hyperparameters with `GridSearchCV(cv=5)`, and explain model predictions. |
| **`lecture5_SentimentPrediction.pdf`** & **`Lec_16_SVM(11).pdf`** | NLP preprocessing (`unicodedata` NFKD accented char removal, `contractions`, regex cleaning, `ToktokTokenizer`, `nltk` stopwords retaining `'no'`/`'not'`, lemmatization), **Bag of Words (`CountVectorizer`)**, **TF-IDF (`TfidfVectorizer` with `sublinear_tf=True`, `ngram_range=(1,2)`)**, **Gensim `Word2Vec`** + `averaged_word2vec_vectorizer`, **Cosine Similarity**, **`LogisticRegression`**, **Linear SVM (`SGDClassifier`)**, and **Keras `Sequential` DNN**. | **Core NLP Normalization (Section 3.4), Candidate Recommendation Models & Tag Prediction (Sections 4 & 5)**: Clean `title` and `body_text` while preserving technical tokens (`c++`, `c#`, `.net`, `node.js`); compare **BoW**, **TF-IDF**, **Averaged Word2Vec**, and **SentenceTransformer** retrieval models alongside supervised tag classifiers. |
| **`lecture5_SentimentPrediction_LLM.pdf`** | Pretrained representation/embedding models using `sentence-transformers` (`SentenceTransformer`), dense semantic embeddings, two-step embedding + lightweight classifier. | **Semantic Question Recommender (Sections 4 & 5)**: Encode questions into dense semantic vectors using `SentenceTransformer('all-MiniLM-L6-v2')` so queries match semantically similar questions even when exact keywords differ. |
| **`Lecture4-Hyperparameters_ModelInterpret_Deployment.pdf`** | `GridSearchCV` / `RandomizedSearchCV`, model interpretability, **Model Persistence (`joblib.dump` / `pickle`)**, and **Streamlit Deployment (`app.py`, `prediction.py`, `requirements.txt`)**. | **Evaluation, Explainability & Deployment (Sections 5 & 6)**: Tune models, interpret recommendations (shared TF-IDF terms + semantic similarity + quality breakdown), export artifacts via `joblib.dump`, and deploy via Streamlit. |

---

## 2. Notebook Structure & Implementation Status

### ✅ Completed & Reviewed (Sections 1 to 3)

#### Section 1: Project Definition & Dataset Sufficiency Justification (Task 1.1 — 5 Marks)
- **1.1 Product Definition & Usefulness**: Defines the end-to-end **Stack Overflow Question Recommendation & Tag Suggestion System** (semantic + lexical question retrieval, automated technical tag suggestion, and quality-aware re-ranking).
- **1.2 Dataset Loading & Sufficiency Table**: Loads `questions_2026.csv.zip` (**20,142 questions**, 13 columns, `10,027` unique tags, `73.2%` with code blocks, `46.1%` answered, `25.4%` accepted, `808` duplicate links).
- **Analysis of Section 1 Results**: In-depth justification of sample volume, multi-modal text/code richness, and ground-truth labels.

#### Section 2: Exploratory Data Analysis (EDA) & Data Cleaning (Task 1.2 & 1.3)
- **2.1 & 2.2 Missing Values, Imputation & Type Casting**: Evaluates missingness across all 13 attributes, distinguishes structural missingness (`closed_reason` `89.27%`, `duplicate_of` `95.99%`, `code_text` `26.77%`) from accidental missingness (`body_text` `0.02%`), casts `creation_date` to UTC timestamps, and derives `resolution_status` (`Accepted Answer` `25.43%`, `Answered (Unaccepted)` `48.88%`, `Unanswered` `25.69%`).
- **2.3 Descriptive Statistics & IQR Outlier Analysis**: Computes summary statistics and $1.5 \times IQR$ bounds (`score` `19.90%` outliers, `view_count` `7.13%` outliers), critically justifying why viral Stack Overflow questions must be retained and log-transformed/robust-scaled rather than deleted.
- **2.4 Tag Distribution & Meta-Tag Discovery**: Visualizes Top 20 tags and resolution status, identifying cross-cutting meta-tags (`advice`, `best-practices`, `tooling-recommendation`) that must be separated from concrete programming language tags.
- **2.5 One-Way ANOVA F-Tests (`scipy.stats.f_oneway`) & Boxplots**: Proves statistically significant differences ($p < 0.01$) across resolution categories for `score` ($F = 366.97$), `has_code` ($F = 342.19$), `view_count` ($F = 99.83$), `body_word_count` ($F = 51.85$), and `title_word_count` ($F = 4.88$).
- **2.6 Correlation Matrix Heatmap**: Demonstrates that log-transforming views (`view_count_log`) nearly doubles correlation with `answer_count` ($r = 0.22 \rightarrow 0.40$) and quadruples correlation with `is_answered` ($r = 0.08 \rightarrow 0.29$), with no severe multicollinearity ($r < 0.65$).

#### Section 3: Data Preprocessing, Feature Engineering & Selection (Task 1.3 — 3 Marks)
- **3.1 Temporal Feature Evaluation, Selection & Numeric Engineering**:
  - Extracts calendar/clock features (`Month`, `Day`, `DayOfWeek`, `Hour`, `TimeOfDayBin`) alongside **Question Recency (`question_age_days`)** and **Age-Normalized View Rate (`views_per_day`, `views_per_day_log`)**.
  - Empirically proves that `Hour` ($r = -0.0014$) and `DayOfWeek` ($r = 0.0067$) have near-zero correlation with engagement and no semantic relevance to user queries, explicitly dropping `['Month', 'Day', 'DayOfWeek', 'Hour', 'TimeOfDayBin']` while retaining `question_age_days` and `views_per_day_log`.
  - Applies `Binarizer`, `np.log1p`, `PolynomialFeatures` interaction terms, fixed-width length binning (`pd.cut`), and adaptive quartile binning (`pd.qcut`).
- **3.2 Feature Scaling & Composite Question Quality Score**:
  - Compares `StandardScaler`, `MinMaxScaler`, and `RobustScaler`, and constructs a bounded $[0, 1]$ age-adjusted **`quality_score`** (`0.15 * view_count_log + 0.15 * views_per_day_log + 0.30 * score_log + 0.15 * answer_count_log + 0.15 * is_answered + 0.10 * has_accepted_answer`).
- **3.3 Categorical Encoding & Primary Domain Tag Extraction**:
  - Extracts the Top 10 technical programming domains (`python`, `c++`, `java`, `javascript`, `c#`, `android`, `c`, `r`, `html`, `ios` — covering `46.61%` of all questions) and applies `LabelEncoder` and `OneHotEncoder`.
- **3.4 NLP Text Normalization Pipeline**:
  - Applies `remove_accented_chars` (`NFKD`), `expand_text_contractions`, custom `remove_special_characters_technical` (preserving `c++`, `c#`, `.net`, `node.js`), `ToktokTokenizer` stopword removal (retaining `'no'` and `'not'`), and noun+verb `WordNetLemmatizer`.
  - Quantifies vocabulary compression (**`-44.9%` total tokens**, **`-52.8%` unique vocabulary size**) and saves `dataset/questions_2026_processed.pkl`.

---

### ⏳ Next Steps to Implement (Sections 4 to 6)

#### Section 4: Candidate Machine Learning & AI Models (Task 1.3 — 3 Marks)
Every model will include theoretical justification, mathematical formulation, suitability analysis for Stack Overflow recommendation, and an in-depth results analysis block:

- **Part A — Candidate Question Retrieval & Recommendation Models**:
  1. **Model 1: Bag-of-Words (`CountVectorizer`, unigrams+bigrams) + Cosine Similarity** (*Lecture 5 Slide 19*): Baseline sparse lexical matching.
  2. **Model 2: Sublinear TF-IDF (`TfidfVectorizer`, `sublinear_tf=True, ngram_range=(1,2)`) + Cosine Similarity** (*Lecture 5 Slide 20*): Downweights common words and emphasizes discriminative technical identifiers.
  3. **Model 3: Gensim `Word2Vec` + `averaged_word2vec_vectorizer` + Cosine Similarity** (*Lecture 5 Slides 26–28*): Dense 300-dimensional CBOW/Skip-gram word vectors trained directly on our tokenized Stack Overflow corpus.
  4. **Model 4: Pretrained Transformer Embeddings (`SentenceTransformer('all-MiniLM-L6-v2')`) + Cosine Similarity** (*Lecture 5 LLM Slides 20–23*): Deep contextual embeddings capturing semantic equivalence across differently worded queries.
  5. **Model 5: Hybrid Quality-Aware Recommender (`TF-IDF + SentenceTransformer + quality_score`) + Unsupervised `KMeans` Topic Clustering** (*Tutorial 3*): Combines exact technical keyword precision (TF-IDF) with deep semantic similarity (`SentenceTransformer`) and community quality (`quality_score`), plus `KMeans` clustering evaluated with `silhouette_score` and visualized in 2D/3D via `PCA` (*Lecture 2 Slides 50–55*).

- **Part B — Candidate Supervised Technical Tag Classification Models**:
  - Train and compare 4 supervised models to predict the primary programming tag for new queries:
    1. **Logistic Regression (`LogisticRegression`)** (*Lecture 5 Slide 22 & Tutorial 4 Task 2*)
    2. **Linear Support Vector Machine (`SGDClassifier(loss='hinge')` / `LinearSVC`)** (*Lecture 5 Slide 22 & SVM Lecture*)
    3. **Random Forest Classifier (`RandomForestClassifier`)** (*Tutorial 4 Task 3.4*)
    4. **Deep Neural Network (`Sequential` with `Dense` and `Dropout` layers)** (*Lecture 1 Slide 42 & Lecture 5 Slides 23–24*)

#### Section 5: Model Training, Hyperparameter Tuning, Evaluation & Selection for Deployment (Task 1.3 — 9 Marks)
- **5.1 Quantitative Evaluation of Retrieval / Recommendation Models**:
  - Benchmark all 5 candidate recommendation models on two objective tasks:
    1. **Duplicate Question Retrieval Benchmark**: Given duplicate questions in the dataset whose canonical target is also present, and held-out **Title-to-Body Self-Retrieval** ($1,000$ test queries), measure **Recall@K / Hit-Rate@K ($K=1, 5, 10$)** and **Mean Reciprocal Rank (MRR)**.
    2. **Tag-Consistency Precision@K ($K=5, 10$)**: Measure the fraction of top-$K$ recommended questions that share at least one ground-truth technical tag with the query question.
    3. **Inference Latency (ms/query) & Memory Footprint (MB)**.
- **5.2 Supervised Tag Classifier Training, Hyperparameter Tuning (`GridSearchCV`) & Evaluation**:
  - `80/20` stratified train/test split (`train_test_split`), 5-fold cross-validation tuning (`GridSearchCV`, *Lecture 4 Slide 6 & Tutorial 4 Task 3.5*), `classification_report` (Accuracy, Macro/Weighted Precision, Recall, F1-Score), and `confusion_matrix` heatmaps.
- **5.3 Model Interpretation & Explainability** (*Lecture 4 Slides 13–19 & Tutorial 4 Task 3.7–3.8*):
  - Global interpretation: Top predictive terms/features per programming tag class.
  - Local recommendation explanation: Break down why a specific question was recommended for a sample query (showing exact matching TF-IDF terms, semantic similarity score, quality score contribution, and predicted tag probabilities).
- **5.4 Critical Comparison & Justification of Model Chosen for Deployment (6 Marks)**:
  - Side-by-side comparison table and multi-metric plots (MRR, Hit@5, Tag Precision@5, Macro-F1 vs. Query Latency and RAM), explicitly justifying why the **Hybrid Recommender (TF-IDF + SentenceTransformer + Quality Re-ranking) paired with the tuned Linear SVM / Logistic Regression Tag Predictor** is optimal for real-world deployment.

#### Section 6: Model Persistence & Deployment Preparation (Task 1.4 — 5 Marks)
- **6.1 Model Persistence (`joblib.dump`)**: Export the fitted TF-IDF vectorizer, precomputed normalized matrices/embeddings, trained tag classifier, `LabelEncoder`, and metadata index into `models/` (*Lecture 4 Slide 24 & 30*).
- **6.2 End-to-End Inference Demonstration**: Run realistic developer queries through `recommend_questions(query_title, query_body, top_k=5)` inside the notebook with rich formatted outputs.
- **6.3 Streamlit Web Application (`app.py`, `prediction.py`, `requirements.txt`)**: Build the interactive Streamlit web application (*Lecture 4 Slides 29–32*) allowing users to enter a question, view predicted tags with confidence bars, inspect the top-$K$ recommended Stack Overflow questions with direct links, filter by resolution status, and view local explainability breakdowns.
