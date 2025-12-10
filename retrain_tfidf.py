import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths
DATA_DIR = "data/imdb"
ART_DIR = "artifacts/tfidf"

os.makedirs(ART_DIR, exist_ok=True)

print("Loading data...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

X = train_df["text"]
y = train_df["label"]

print("Splitting train/val...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42,
    stratify=y,
)

print("Fitting TF-IDF vectorizer...")
vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_tr = vect.fit_transform(X_train)
X_val_t = vect.transform(X_val)
X_test_t = vect.transform(test_df["text"])

print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=500, n_jobs=-1)
lr.fit(X_tr, y_train)
print("LR validation report:")
print(classification_report(y_val, lr.predict(X_val_t)))

print("Saving LR + TF-IDF to artifacts/tfidf/lr_tfidf.joblib")
joblib.dump(
    {"model": lr, "vectorizer": vect},
    os.path.join(ART_DIR, "lr_tfidf.joblib"),
)

print("Training Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_tr, y_train)
print("NB validation report:")
print(classification_report(y_val, nb.predict(X_val_t)))

print("Saving NB + TF-IDF to artifacts/tfidf/nb_tfidf.joblib")
joblib.dump(
    {"model": nb, "vectorizer": vect},
    os.path.join(ART_DIR, "nb_tfidf.joblib"),
)

print("Done. Artifacts written to:", ART_DIR)
