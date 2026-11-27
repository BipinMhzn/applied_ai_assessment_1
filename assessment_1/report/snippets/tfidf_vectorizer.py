def make_vectorizer(word_ngrams, char_ngrams, min_df, max_features, stop_words):
    word_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=word_ngrams, min_df=min_df,
                                      max_features=max_features, sublinear_tf=True, stop_words=stop_words)
    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngrams, min_df=min_df,
                                      max_features=max_features, sublinear_tf=True)
    return FeatureUnion([("word", word_vectorizer), ("char", char_vectorizer)])
