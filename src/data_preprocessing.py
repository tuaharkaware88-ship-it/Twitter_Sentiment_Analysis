import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


short_dict = {'u': 'you', 'r': 'are', 'n': 'and', 'lol': 'laughing out loud', 'lt3': 'love'}
lemmatizer = WordNetLemmatizer()


def preprocess_tweet(text, stopwords_set):
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters (keep letters, numbers, spaces)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Reduce repeated letters (coooool -> coool)
    text = ' '.join([word for word in text.split() if word not in stopwords_set])
    text = ' '.join([short_dict.get(word, word) for word in text.split()])
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text


def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def get_train_test(csv_path=None, test_size=0.2, random_state=42):
    """Load CSV, preprocess text and return X_train, X_test, y_train, y_test.

    csv_path: optional path to the csv. If None, default to the repository's `data/` file.
    """

    if csv_path is None:
        # default: project root is parent of src
        csv_path = r"C:\Sentiment_Analysis_Project\data\twitter_sentiment_Analysis_data.csv"

    df = pd.read_csv(csv_path)

    # map target (original code used 4 as positive)
    df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

    # drop duplicates as before
    df = df.drop_duplicates(subset=['text'], keep='first').reset_index(drop=True)

    # Handles Nan values in 'text' column:
    df = df.fillna({'text': ''})

    stopwords_set = set(stopwords.words('english'))

    # preprocess
    print("Preprocessing text...")

    df['text'] = df['text'].apply(lambda x: preprocess_tweet(x, stopwords_set))

    # lemmatize
    print("Lemmatizing text...")

    df['text'] = df['text'].apply(lemmatize_text)

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=test_size, random_state=random_state)

    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Save processed data

    os.makedirs(r"C:\Sentiment_Analysis_Project\data\processed", exist_ok=True)

    X_train.to_csv(r"C:\Sentiment_Analysis_Project\data\processed\X_train.csv", index=False)
    X_test.to_csv(r"C:\Sentiment_Analysis_Project\data\processed\X_test.csv", index=False)
    y_train.to_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_train.csv", index=False)
    y_test.to_csv(r"C:\Sentiment_Analysis_Project\data\processed\y_test.csv", index=False)

    print("data loaded from processed files.")


