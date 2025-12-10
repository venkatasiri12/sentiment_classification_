import streamlit as st
import joblib, os, pickle
from tensorflow.keras.models import load_model
st.title('Sentiment Classifier Demo')
tf_art = 'artifacts/tfidf/lr_tfidf.joblib'
if os.path.exists(tf_art):
    pkg = joblib.load(tf_art)
    tf_model = pkg['model']; tf_vect = pkg['vectorizer']
else:
    tf_model=tf_vect=None
nn_tok='artifacts/nn/tokenizer.pkl'; nn_model='artifacts/nn/lstm.h5'
if os.path.exists(nn_tok) and os.path.exists(nn_model):
    with open(nn_tok,'rb') as f: tokenizer=pickle.load(f)
    nn = load_model(nn_model)
else:
    tokenizer=nn=None
text = st.text_area('Enter a review')
if st.button('Predict'):
    if tf_model:
        p = int(tf_model.predict(tf_vect.transform([text]))[0])
        st.write('TF-IDF LR label:', p)
    if nn is not None:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=256)
        st.write('NN prob:', float(nn.predict(seq)[0][0]))
