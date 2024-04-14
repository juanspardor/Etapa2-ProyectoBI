from joblib import load
from RequiredFunctions import preprocessing_text, apply_preprossesing, remove_non_ascii, to_lowercase, contains_number, remove_numbers, remove_punctuation, remove_stopwords

from TransformerClass import Transformer_Representacion_Seleccion

class Model:

    def __init__(self):
        self.model = load("assets/pipeline.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
