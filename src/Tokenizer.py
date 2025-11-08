from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os
import numpy as np
import joblib
from joblib import dump


def padding_text(vocab_size, max_len):

    X_train = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\X_train.csv")["text"].fillna('')
    X_test = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\X_test.csv")["text"].fillna('')
    y_train = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_train.csv")["target"]
    y_test = pd.read_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_test.csv")["target"]

    print("Data loaded.")

    # Convert DataFrames to list of strings
    X_train_texts = X_train.squeeze().astype(str).tolist()
    X_test_texts = X_test.squeeze().astype(str).tolist()

    # 1. Creating tokenizer:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<oov>") 

    print("Tokenizer created.") 
    # 2. fitting on training data:
    tokenizer.fit_on_texts(X_train_texts)

    print("Tokenizer fitted on training data.") 
    # 3. Text to integer conversion:
    X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
    X_test_seq  = tokenizer.texts_to_sequences(X_test_texts)

    # Padding: to maintain the equal length for the input
    # padding only happend if the texts are in list and integer(list of integers) and for that tokenizer helps.
    print("Converting texts to sequences and padding...")
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    print("Padding completed.") 

    # Make a directory for padded data
    padded_dir = r"C:\Sentiment_Analysis_Project\data\padded_data"
    os.makedirs(padded_dir, exist_ok=True)


    # Save padded data as .npy files (fast & memory-efficient)
    np.save(os.path.join(padded_dir, "X_train_padded.npy"), X_train_padded)
    np.save(os.path.join(padded_dir, "X_test_padded.npy"), X_test_padded)
    print(f"Padded data saved successfully in: {padded_dir}")



    # Save tokenizer.
    tokenizer_path = r"C:\Sentiment_Analysis_Project\models\rnn"
    os.makedirs(tokenizer_path, exist_ok=True)
    dump(tokenizer, os.path.join(tokenizer_path, "tokenizer.joblib"))
    print(f"Tokenizer saved successfully in: {tokenizer_path}")


    return X_train_padded, X_test_padded, y_train, y_test


if __name__ == "__main__":
    X_train_padded, X_test_padded, y_train, y_test = padding_text(vocab_size=10000, max_len=100)