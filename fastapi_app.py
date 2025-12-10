from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os
app = FastAPI()
class Item(BaseModel): text: str
tf_art = 'artifacts/tfidf/lr_tfidf.joblib'
tf_pkg = joblib.load(tf_art) if os.path.exists(tf_art) else None
@app.post('/predict/tfidf')
def predict(item: Item):
    if tf_pkg is None: return {'error':'no model'}
    model = tf_pkg['model']; vect = tf_pkg['vectorizer']
    return {'label': int(model.predict(vect.transform([item.text]))[0])}
