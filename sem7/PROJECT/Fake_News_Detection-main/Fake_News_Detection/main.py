
import numpy as np
import pandas as pd

import keras
from keras import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
#import nltk
#nltk.download('stopwords')

import re
from string import punctuation

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from keras.models import load_model

import pickle


# Define a flask app
app = Flask(__name__)

# Load your trained model
model = load_model('model.h5')

#stop_words = stopwords.words("english")
#using below command for now because above one is causing error..below list is same as above one.
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
              'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
              'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
              'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
              'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
              'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
              'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
              'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
              'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
              'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
              'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
              'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
              "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
              "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
              "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
ps = PorterStemmer()

@app.route('/')
def home():
    return render_template('index.html')


def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    text = ' '.join(text)
    text = ''.join(p for p in text if p not in punctuation)

    return text



@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    string = request.form.get("News")
    string = preprocess(string)
    string = [string]

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    str_sequences = tokenizer.texts_to_sequences(string)
    padded_str = pad_sequences(str_sequences, maxlen=40, truncating='post', padding='post')
    result = model.predict_classes(padded_str)
    result = int(result.flatten())
    acc = model.predict(padded_str)

    acc = float(acc.flatten())
    # print(acc)
    acc = acc * 100
    acc = round(acc, 2)
    # print(acc)
    if result == 0:
        return render_template('index.html',
                               prediction_text='This news can be Real.\n Probability of this news being real is {}%'.format(100-acc))
    else:
        return render_template('index.html',
                               prediction_text='This news can be Fake.\n Probability of this news being fake is {}%'.format(acc))


if __name__ == '__main__':
    app.run(debug=True)
