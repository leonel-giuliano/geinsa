import re
import joblib
import pandas as pd
import firebase_admin

from firebase_admin import credentials
from firebase_admin import firestore
from sentence_transformers import util

import config
from client import *

UID_PRED = 0

SCORE_IX = 1
MODEL_SCORE_IX = 2


def strnormalize(s):
    rep_accent = str.maketrans("áéíóúü", "aeiouu")

    # Normalize the text
    s = s.lower().translate(rep_accent)
    s = re.sub(r"[^a-zñ\s]", "", s)

    return s.split()


def tokenize(df, cols, join=False) -> pd.Series:
    tk = df[cols].fillna("")
    tk = tk.agg(" ".join, axis=1).apply(strnormalize)

    # Returns the Series with all the words in the article
    return tk if not join else tk.apply(" ".join)


def semantic_rank(query, embedder, article_embeddings, top_k=50):
    # Get the embeddings of the query used in order to get similar words
    query_emb = embedder.encode(query, convert_to_tensor=True)

    # Transform the embeddings into actual scores
    scores = {aid: float(util.cos_sim(query_emb, emb))
              for aid, emb in article_embeddings.items()}

    # Sort by similarity
    # Saves only the 'k' num of first items
    ranked = sorted(scores.items(), key=lambda x: -x[SCORE_IX])[:top_k]
    return ranked


def recommend(query, model, embedder, article_embeddings, aid_map, item_features, top_k=50):
    # Get the articles thar are going to be ranked
    candidates = semantic_rank(query, embedder, article_embeddings, top_k)
    article_ids = [aid for aid, _ in candidates]

    aids = [aid_map[aid] for aid in article_ids]
    scores = model.predict(UID_PRED, aids, item_features=item_features)

    # Combine semantic + LightFM scores (weighted)
    result = [(aid, sem_score, model_score)
              for (aid, sem_score), model_score in zip(candidates, scores)]

    # Rerank the recommendations prioritizing the model
    ranked = sorted(result, key=lambda x: -x[MODEL_SCORE_IX])
    return ranked


def main():
    # Initiate communiaction with the database
    cred = credentials.Certificate("../firebase-cred.json")
    firebase_admin.initialize_app(cred)

    db = firestore.client()
    coll_ref = db.collection("busquedas")

    # Get the model data
    data = joblib.load("model-data.pkl")

    config.model = data["model"]
    config.item_features = data["item_features"]
    ds = data["dataset"]
    config.embedder = data["embedder"]
    config.article_embeddings = data["article_embeddings"]

    config.articles = pd.read_csv("/content/drive/MyDrive/Geinsa/data/noticias_econojournal_completo.csv")
    config.articles = config.articles.rename(columns={
        "ID": "id",
        "Título": "title",
        "Descripción": "description",
        "Cuerpo": "body"
    })

    _, _, config.aid_map, _ = ds.mapping()

    coll_ref.on_snapshot(on_snapshot)

    print("Listening...")
    while True:
        pass


if __name__ == "__main__":
    main()
