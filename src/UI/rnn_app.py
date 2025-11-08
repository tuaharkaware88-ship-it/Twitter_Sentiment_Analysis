import streamlit as st
import joblib
from nltk.corpus import stopwords
from data_preprocessing import preprocess_tweet, lemmatize_text
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
# ----------------------------------------------------------
# 1️⃣ Load necessary stuff
# ----------------------------------------------------------
stopwords_set = set(stopwords.words('english'))

# Load your saved model and vectorizer
model = load_model(r"C:\Sentiment_Analysis_Project\models\rnn\rnn_model.h5")
tokenizer = load(r"C:\Sentiment_Analysis_Project\models\rnn\tokenizer.joblib")

# Model was trained with sequences padded to length 100
MAX_SEQ_LEN = 100

# ----------------------------------------------------------
# 2️⃣ Function to predict sentiment
# ----------------------------------------------------------
def predict_sentiment(text):
    # clean the text same as training
    clean_text = preprocess_tweet(text, stopwords_set)
    clean_text = lemmatize_text(clean_text)

    # convert text to sequence and pad (tokenizer is a Keras Tokenizer saved with joblib)
    seq = tokenizer.texts_to_sequences([clean_text])
    vector_data = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

    # make prediction (model outputs probability via sigmoid)
    pred_prob = model.predict(vector_data)[0][0]

    # threshold probability at 0.5
    if pred_prob >= 0.5:
        return f"😊 Positive Sentiment"
    else:
        return f"😞 Negative Sentiment"


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
