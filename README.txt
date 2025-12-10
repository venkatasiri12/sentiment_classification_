# Sentiment Classifier Demo

This project is a sentiment analysis web app for movie reviews, built with:

- **TF-IDF + Logistic Regression** (classical ML)
- **LSTM Neural Network** (deep learning with Keras/TensorFlow)
- **Streamlit** for the web UI

Enter a review and see both a discrete label (0/1) and a neural network probability.

---

## Project Structure

```text
sentiment_project/
│
├── app.py                     # Streamlit application
├── retrain_tfidf.py           # Script to retrain TF-IDF models
├── requirements.txt           # Python dependencies
│
├── artifacts/
│   ├── tfidf/
│   │   ├── lr_tfidf.joblib    # LR model + fitted TF-IDF vectorizer
│   │   └── nb_tfidf.joblib    # NB model + TF-IDF vectorizer (optional)
│   └── nn/
│       ├── tokenizer.pkl      # Tokenizer for LSTM
│       └── lstm.h5            # Trained LSTM model
│
└── data/
    └── imdb/
        ├── train.csv          # Training data
        └── test.csv           # Test data

