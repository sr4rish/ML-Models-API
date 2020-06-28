from flask import Flask, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Loading model to compare the results
model = pickle.load(open('restaurantReview/model.pkl','rb'))
cv = pickle.load(open('restaurantReview/transform.pkl','rb'))
ps = PorterStemmer()
nltk.download('stopwords')

def sentiment(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.split()
    review = [ps.stem(words) for words in review if not words in set(stopwords.words('english'))]
    review = ' '.join(review)
    
    review = [review]
    review = cv.transform(review).toarray()
    result = model.predict(review)
    return result

@app.route('/predict/', methods=['POST'])
def predict():
    review = request.json['review']
    result1 = sentiment(review)
    if result1[0] == 0:
        result = 'negative'
    else:
        result = 'positive'
    response = jsonify(result)
    return response


if __name__ == "__main__":
    app.run()