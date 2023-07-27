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
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter("error", InconsistentVersionWarning)
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
        try:
           vec = joblib.load('vectorizer.pkl')
        except InconsistentVersionWarning as w:
           print(w.original_sklearn_version)
        
        pre_num = vec.transform(pre)
        return pre_num

class predict_input:
    def get_prediction(input_data):
        
        input = Preprocessing.preprocess_data(input_data)

        # Load the model from the file
        try:
           classifier = joblib.load('model.pkl')
        except InconsistentVersionWarning as w:
           print(w.original_sklearn_version)
        
        # Use the loaded model to make predictions
        prediction = classifier.predict(input)

        return prediction


