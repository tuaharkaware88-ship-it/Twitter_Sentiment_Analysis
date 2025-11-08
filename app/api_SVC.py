from flask import Flask, jsonify, request
import joblib
from nltk.corpus import stopwords
from flask_cors import CORS
from data_preprocessing import preprocess_tweet, lemmatize_text


# create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


stopwords_set = set(stopwords.words('english'))

# Load your saved model and vectorizer
model = joblib.load(r"C:\Sentiment_Analysis_Project\models\svc\svm_tfidf.joblib")
vectorizer = joblib.load(r"C:\Sentiment_Analysis_Project\models\svc\tfidf_vectorizer.joblib")


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


# Home route
@app.route('/')
def home():
    return "Welcome to My Flask API!"


# Get Sentiment API
@app.route('/get-sentiment', methods=['GET'])
def get_sentiment():
    data = request.get_json()   # assume request will be { "text" : "I love ML."}
    text = data['text']

    sentiment = predict_sentiment(text)

    response = {
        "status": "Success",
        "sentiment": sentiment
    }

    return jsonify(response)




# Run the app
if __name__ == '__main__':
    app.run(debug=True)
