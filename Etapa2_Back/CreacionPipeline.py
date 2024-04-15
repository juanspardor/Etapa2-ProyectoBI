 # librería Natural Language Toolkit, usada para trabajar con textos
import nltk
# Punkt permite separar un texto en frases.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


import pandas as pd
import numpy as np
import sys

import re, string, unicodedata
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix

from scipy import stats as st

from nltk.stem import SnowballStemmer

from statistics import mode

from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib.pyplot as plt
import re

import json
from sklearn.preprocessing import FunctionTransformer
import joblib

#Cargamos los datos
data=pd.read_csv("assets/tipo2_entrenamiento_estudiantes.csv", sep=',', encoding = 'utf-8')


## TODAS LAS COSAS QUE SE NECESITAN
#Ahora creamos las funciones que nuestro pipe podria necesitar
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
          new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
          new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
            new_word = word.lower()
            if new_word != ' ':
                new_words.append(new_word)
    return new_words

def contains_number(s):
    # Regular expression pattern to match any digit
    pattern = re.compile(r'\d')
    return bool(pattern.search(s))


def remove_numbers(words):
    new_words = []
    for word in words:
        if not contains_number(word):
            new_words.append(word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        if word is not None:
            # Adjusted regular expression pattern to exclude colon
            new_word = re.sub(r'[^\w\s:]', '', word)
            if new_word != '':
                new_words.append(new_word)
    return new_words


spanish_stopwords = set(stopwords.words('spanish'))
def remove_stopwords(words):
    new_words = []
    for word in words:
        if word is not None:
            if word not in spanish_stopwords:
                new_words.append(word)
    return new_words

def apply_preprossesing(words):
    words = to_lowercase(words)
    words = remove_numbers(words)
    words = remove_punctuation(words)
   # words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words

#Esta version elimina convierte a misnusculas, quita puntuacion, quita stopwords
def preprocessing_text(texto):

    texto['Review'] = texto['Review'].apply(contractions.fix)
    texto['words'] = texto['Review'].apply(word_tokenize)
    texto['words'] = texto['words'].apply(apply_preprossesing)
    texto['words'] = texto['words'].apply(lambda x: ' '.join(map(str, x)))
    return texto['words']

#Necesitamos un transformador custom para que haga la representacion y solo utilice las columnas que queremos
class Transformer_Representacion_Seleccion:
    def __init__(self, count_vectorizer):
        self.count_vectorizer = count_vectorizer
        self.palabras = None
        with open('assets/palabras_utiles.json', 'r') as f:
            lista_cargada = json.load(f)
            
        self.palabras_deseadas = pd.Index(lista_cargada)
    
    def fit(self, X, y=None):
        X_transformed = self.count_vectorizer.fit_transform(X)
        self.palabras = self.count_vectorizer.get_feature_names_out()
        return self
    
    def transform(self, X):
        # Transform data using CountVectorizer
        X_transformed = self.count_vectorizer.transform(X)
        # Get feature names from CountVectorizer
        X_todas_palabras = pd.DataFrame(X_transformed.toarray(), columns=self.palabras)
        
        return X_todas_palabras[self.palabras_deseadas].copy()
    
#Creamos el pipe (solo probando la parte de preprocesamiento)
#pipe = Pipeline([
   # ('preprocess', FunctionTransformer(preprocessing_text)),
  #  ('representacion', Transformer_Representacion_Seleccion(CountVectorizer())),
 #   ('clf', MultinomialNB(alpha = 0.92)),
#])

#pipe.fit(data, data['Class'])

#joblib.dump(pipe, 'assets/pipeline_funcional.joblib')

