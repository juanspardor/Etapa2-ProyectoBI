#Necesitamos un transformador custom para que haga la representacion y solo utilice las columnas que queremos
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

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