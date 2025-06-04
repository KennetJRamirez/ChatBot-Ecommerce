from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os

app = FastAPI()

DATABASE_URL = "postgresql://postgres:GWxecCCNwRQhCYWpzlMfPuNJXJkjKTBW@turntable.proxy.rlwy.net:32870/railway"
engine = create_engine(DATABASE_URL)

modelo = SentenceTransformer('distiluse-base-multilingual-cased-v2')

class Query(BaseModel):
    texto: str

df = None
embeddings = None

@app.on_event("startup")
def cargar_datos():
    global df, embeddings
    print("Cargando productos desde DB...")
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT product_id, product_name, description, price FROM productos"), conn)
    df['texto'] = df['product_name'] + ' ' + df['description']

    if os.path.exists("embeddings.pt"):
        print("Cargando embeddings desde disco...")
        embeddings = torch.load("embeddings.pt")
    else:
        print("Generando embeddings...")
        embeddings = modelo.encode(df['texto'].tolist(), convert_to_tensor=True)
        torch.save(embeddings, "embeddings.pt")

    print("Backend listo.")

@app.post("/recomendar")
def recomendar(query: Query):
    query_embedding = modelo.encode(query.texto, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_indices = torch.topk(scores, k=5).indices
    resultados = df.iloc[top_indices.cpu().numpy()]
    return resultados[['product_name', 'price', 'description']].to_dict(orient="records")
