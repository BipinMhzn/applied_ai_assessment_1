embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# embeddings: train catalog, test queries, val queries, full catalog (used by the app)
train_embeddings = embedding_model.encode(train_df["text_all"].tolist(), convert_to_numpy=True, show_progress_bar=True)
test_embeddings = embedding_model.encode(test_df.iloc[:num_eval_queries]["text_all"].tolist(), convert_to_numpy=True, show_progress_bar=True)
val_embeddings = embedding_model.encode(val_df["text_all"].tolist(), convert_to_numpy=True, show_progress_bar=True)
all_embeddings = embedding_model.encode(questions_df["text_all"].tolist(), convert_to_numpy=True, show_progress_bar=True)

embedding_similarity_matrix = cosine_similarity(test_embeddings, train_embeddings)
