from typing import Optional
from DataModel import DataModel
from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
from joblib import load 
from nbconvert import PythonExporter
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pickle
from typing import List
import CreacionPipeline
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Esto permite todos los orígenes, ajusta según sea necesario
    allow_credentials=True,
    allow_methods=["*"],  # Esto permite todos los métodos, ajusta según sea necesario
    allow_headers=["*"],
)

@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}


@app.post("/predict")
def make_predictions(datdf_data: DataModel):
   df = pd.DataFrame(datdf_data.dict())
   df.columns = datdf_data.columns()
   model = load("assets/pipeline_funcional.joblib")
   result = model.predict(df)
   result_list = result.tolist()

   return JSONResponse(content={"predictions": result_list})

@app.post("/predict_probs")
def make_predictions(datdf_data: DataModel):
   df = pd.DataFrame(datdf_data.dict())
   df.columns = datdf_data.columns()
   model = load("assets/pipeline_funcional.joblib")
   result = model.predict(df)
   predicciones = result.tolist()

   probabilidades = model.predict_proba(df)
   probas_finales = []
   for probabilidad in probabilidades:
      probas_finales.append(max(probabilidad))

   return JSONResponse(content={"predictions": predicciones, "probabilities": probas_finales})
