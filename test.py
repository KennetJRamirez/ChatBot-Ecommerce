from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware
import torch
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = "postgresql://postgres:GWxecCCNwRQhCYWpzlMfPuNJXJkjKTBW@turntable.proxy.rlwy.net:32870/railway"
engine = create_engine(DATABASE_URL)
modelo = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

class Query(BaseModel):
    texto: str
    k: int = 5  # Cantidad de productos a mostrar
df = None
embeddings = None

@app.on_event("startup")
def cargar_datos():
    global df, embeddings
    print("Cargando productos desde DB...")
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT product_id, product_name, product_brand, gender, price, description, primary_color FROM productos"), conn)
        df['texto'] = (
            df['product_brand'].fillna('') + ' ' +
            df['product_name'].fillna('') + '. ' +
            df['gender'].fillna('') + '. ' +
            df['primary_color'].fillna('') + '. ' +
            df['description'].fillna('')
        ).str.strip()

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
    global df, embeddings
    if embeddings is None or df is None:
        raise HTTPException(status_code=503, detail="Datos no cargados aún")

    if query.k <= 0:
        raise HTTPException(status_code=400, detail="El parámetro k debe ser positivo")

    query_embedding = modelo.encode(query.texto, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]

    k = min(query.k, len(df))  # No pedir mas de lo indicado
    top_indices = torch.topk(scores, k=k).indices
    resultados = df.iloc[top_indices.cpu().numpy()]

    similitudes = scores[top_indices].cpu().tolist()

    response = []
    for i, row in resultados.iterrows():
        response.append({
            "product_name": row['product_name'],
            "price": row['price'],
            "description": row['description'],
            "similarity_score": round(similitudes.pop(0), 4)
        })

    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
