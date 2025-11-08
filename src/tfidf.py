from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import dump
import pandas as pd
import os
import scipy.sparse as sp


def get_tfidf_data():

    print("Generating TF-IDF data...")

    X_train = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\X_train.csv")["text"].fillna('')
    X_test = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\X_test.csv")["text"].fillna('')
    y_train = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_train.csv")["target"]
    y_test = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_test.csv")["target"]


    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))

    tfidf_train = tfidf.fit_transform(X_train)
    tfidf_test = tfidf.transform(X_test)

    print(f"Number of features for TF-IDF: {len(tfidf.get_feature_names_out())}")

    # Make a directory to save TF-IDF data.
    os.makedirs(r"C:\Sentiment_Analysis_Project\data\tfidf_data", exist_ok=True)


    # Save sparse matrices: Stores only non-zero values with their positions. "Memory efficient for text data."

    sp.save_npz(r"C:\Sentiment_Analysis_Project\data\tfidf_data\tfidf_train.npz", tfidf_train)
    sp.save_npz(r"C:\Sentiment_Analysis_Project\data\tfidf_data\tfidf_test.npz", tfidf_test)

    print("TF-IDF sparse matrices saved.")

    
    # Save the vectorizer
    dump(tfidf, r"C:\Sentiment_Analysis_Project\models\svc\tfidf_vectorizer.joblib")

    return tfidf_train, tfidf_test, y_train, y_test


if __name__ == "__main__":
    tfidf_train, tfidf_test, y_train, y_test = get_tfidf_data()
    print("TF-IDF data prepared successfully!")







