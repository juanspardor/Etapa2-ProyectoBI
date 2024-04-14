from typing import Optional
from DataModel import DataModel
from pydantic import BaseModel
from PredictionModel import Model
from fastapi import FastAPI
import pandas as pd
from joblib import load 
print("Before import")
from RequiredFunctions import preprocessing_text, apply_preprossesing, remove_non_ascii, to_lowercase, contains_number, remove_numbers, remove_punctuation, remove_stopwords
print("After import")
app = FastAPI()
from TransformerClass import Transformer_Representacion_Seleccion


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.post("/predict")
def make_predictions(dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    print(df)
    df.columns = dataModel.columns()
    modelo = Model()
    result = modelo.predict(df)
    return result
