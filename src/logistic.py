from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from joblib import dump
import scipy.sparse as sp   
import pandas as pd 


def train_test_logistic_with_tfidf():
    
    tfidf_train = sp.load_npz(r"C:\Sentiment_Analysis_Project\data\tfidf_data\tfidf_train.npz")
    tfidf_test = sp.load_npz(r"C:\Sentiment_Analysis_Project\data\tfidf_data\tfidf_test.npz")
    y_train = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_train.csv")["target"]
    y_test = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_test.csv")["target"]


    logistic = LogisticRegression()
    logistic.fit(tfidf_train, y_train)

    print("Predicting with Logistic and TF-IDF features...")

    logistic_tfidf_pred = logistic.predict(tfidf_test)

    # Predictions:

    logistic_tfidf_accuracy = accuracy_score(y_test, logistic_tfidf_pred)

    logistic_classification_report = classification_report(y_test, logistic_tfidf_pred)

    print(f"Logistic with TF-IDF Accuracy: {logistic_tfidf_accuracy:.4f}")
    print(f"Logistic_classification_report:", logistic_classification_report)



    model_path = r"C:\Sentiment_Analysis_Project\models\logistic_regression\logistic_tfidf.joblib"

    dump(logistic, model_path)

    print(f"Logistic tfidf model saved to: {model_path}")




if __name__ == "__main__":
    train_test_logistic_with_tfidf()



