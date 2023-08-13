import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify, render_template
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Your preprocessing code
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review






with open('modelnewl.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


with open('countvectorizer.pkl', 'rb') as cv_file:
    cv = pickle.load(cv_file)

new_text = "This is an example sentence to be predicted."
preprocessed_text = preprocess_text(new_text)


text_vector = cv.transform([preprocessed_text]).toarray()

app = Flask(__name__)

@app.route('/',methods = ['GET'])
def index():
    return jsonify({'message': 'Hello, World!'})


@app.route('/predict', methods=['POST'])
def predict():
     if request.method == 'POST':
        data = request.form['news_text']
        print(data)
        prediction = model.predict(text_vector)
        return jsonify({'message': 'result'+str(prediction)})
     
if __name__ == '__main__':
    app.run()