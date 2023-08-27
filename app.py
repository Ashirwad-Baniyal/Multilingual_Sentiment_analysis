# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from googletrans import Translator

app = Flask(__name__,template_folder='templates')

# Load the sentiment analysis model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
translator = Translator()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        translated_text = translator.translate(text, dest='en')
        text_vectorized = vectorizer.transform([translated_text.text])
        sentiment = model.predict(text_vectorized)[0]
        result = sentiment
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
