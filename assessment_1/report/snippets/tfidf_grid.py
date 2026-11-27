tfidf_param_grid = [
    {"word_ngrams": (1, 1), "char_ngrams": (3, 5), "min_df": 2, "max_features": 100000, "stop_words": None},
    {"word_ngrams": (1, 2), "char_ngrams": (3, 5), "min_df": 2, "max_features": 100000, "stop_words": None},
    {"word_ngrams": (1, 2), "char_ngrams": (3, 5), "min_df": 3, "max_features": 50000, "stop_words": None},
    {"word_ngrams": (1, 2), "char_ngrams": (3, 5), "min_df": 2, "max_features": 100000, "stop_words": "english"},
]
best_val_precision, best_tfidf_params = -1.0, None
for params in tfidf_param_grid:
    val_precision = validation_precision_at_5(make_vectorizer(**params))
    print(f"grid {params} -> val precision@5 = {val_precision:.3f}")
    if val_precision > best_val_precision:
        best_val_precision, best_tfidf_params = val_precision, params
print(f"best TF-IDF params (val precision@5={best_val_precision:.3f}): {best_tfidf_params}")
