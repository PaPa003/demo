import numpy as np
import pandas as pd
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import joblib
import nltk
nltk.download('stopwords')


class Preprocessing:
    def stemmed(content):
        stemmer = PorterStemmer()
        stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content
    
    def preprocess_data(y):
        pre = [Preprocessing.stemmed(y)]
        # Load the model from the file
        vec = joblib.load(r'vectorizer.pkl')
        pre_num = vec.transform(pre)
        return pre_num

class predict_input:
    def get_prediction(input_data):
        input = Preprocessing.preprocess_data(input_data)

        # Load the model from the file
        classifier = joblib.load(r'model.pkl')
        
        # Use the loaded model to make predictions
        prediction = classifier.predict(input)

        return prediction


