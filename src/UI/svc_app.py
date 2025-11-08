import streamlit as st
import joblib
from nltk.corpus import stopwords
from data_preprocessing import preprocess_tweet, lemmatize_text
from joblib import load

# ----------------------------------------------------------
# 1️⃣ Load necessary stuff
# ----------------------------------------------------------
stopwords_set = set(stopwords.words('english'))

# Load your saved model and vectorizer
model = load(r"C:\Sentiment_Analysis_Project\models\svc\svm_tfidf.joblib")
vectorizer = load(r"C:\Sentiment_Analysis_Project\models\svc\tfidf_vectorizer.joblib")

# ----------------------------------------------------------
# 2️⃣ Function to predict sentiment
# ----------------------------------------------------------
def predict_sentiment(text):
    # clean the text same as training
    clean_text = preprocess_tweet(text, stopwords_set)
    clean_text = lemmatize_text(clean_text)

    # convert to TF-IDF numbers
    text_tfidf = vectorizer.transform([clean_text])

    # make prediction
    pred = model.predict(text_tfidf)[0]

    # convert number (0 or 1) to meaning
    if pred == 1:
        return "😊 Positive Sentiment"
    else:
        return "😞 Negative Sentiment"


# ----------------------------------------------------------
# 3️⃣ Streamlit UI
# ----------------------------------------------------------
st.title("🧠 Simple Sentiment Analysis App")
st.write("Type any sentence below and the model will tell if it's Positive or Negative!")

# text box for user input
user_text = st.text_area("Enter your text here:")

# when button is clicked
if st.button("Predict Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner('Analyzing sentiment...'):
            result = predict_sentiment(user_text)
        st.success(result)
