from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from joblib import dump
import scipy.sparse as sp
import pandas as pd


def train_test_svc_with_tfidf():

    tfidf_train = sp.load_npz(r"C:\Sentiment_Analysis_Project\data\tfidf_data\tfidf_train.npz")
    tfidf_test = sp.load_npz(r"C:\Sentiment_Analysis_Project\data\tfidf_data\tfidf_test.npz")
    y_train = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_train.csv")["target"]
    y_test = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_test.csv")["target"]

    svm = LinearSVC()

    # tfidf_train, tfidf_test, y_train, y_test = get_tfidf_data()

    svm.fit(tfidf_train, y_train)

    print("Predicting with SVC and TF-IDF features...")

    svm_tfidf_pred = svm.predict(tfidf_test)

    svm_tfidf_accuracy = accuracy_score(svm_tfidf_pred, y_test)

    print(f"SVC with TF-IDF Accuracy: {svm_tfidf_accuracy:.4f}")

    
    model_path = r"C:\Sentiment_Analysis_Project\models\svc\svm_tfidf.joblib"

    dump(svm, model_path)

    print(f"SVM TF-IDF model saved to: {model_path}")

    return svm, svm_tfidf_accuracy  


if __name__ == "__main__":
    train_test_svc_with_tfidf()



    