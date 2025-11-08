# Sentiment Analysis Project

A text sentiment analysis project featuring three machine learning models:
- Support Vector Machine (SVC) with TF-IDF vectorization
- Logistic Regression with TF-IDF vectorization
- Recurrent Neural Network (RNN) with word embeddings

## Features
- Streamlit web interfaces for interactive prediction
- Flask API for SVC model integration
- Text preprocessing and vectorization
- Trained models saved in `models/` directory

## Files & Components

- `src/UI/` — Streamlit web apps for each model
- `app/api_SVC.py` — Flask API endpoint
- `models/` — Saved model files
- `data/` — Training and processed datasets

## Quick Start (Windows PowerShell):

1. Setup environment:
```powershell
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run Streamlit interfaces:
```powershell
streamlit run src/UI/rnn_app.py     # RNN model
streamlit run src/UI/svc_app.py     # SVC model
streamlit run src/UI/logistic_app.py # Logistic Regression
```

3. Or run Flask API:
```powershell
$env:FLASK_APP='app/api_SVC.py'
flask run
```

## Dependencies
Main packages required:
- streamlit - for web interfaces
- scikit-learn - for SVC and Logistic Regression
- tensorflow - for RNN model
- flask - for API
- nltk - for text processing
- pandas & numpy - for data handling

For full list, see `requirements.txt`
