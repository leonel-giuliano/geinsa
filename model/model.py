import re
import torch
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

SEMANTIC_COLS = ["title", "description", "body"]


def strnormalize(s):
    rep_accent = str.maketrans("áéíóúü", "aeiouu")

    # Normalize the text
    s = s.lower().translate(rep_accent)
    s = re.sub(r"[^a-zñ\s]", "", s)

    return s.split()


def join_strcols(df, cols) -> pd.Series:
    tk = df[cols].fillna("")
    tk = tk.agg(" ".join, axis=1).apply(strnormalize)

    # Returns the Series with all the words in the article
    return tk.apply(" ".join)


def tokenize(df, cols, join=False) -> pd.Series:
    tk = df[cols].fillna("")
    tk = tk.agg(" ".join, axis=1).apply(strnormalize)

    # Returns the Series with all the words in the article
    return tk if not join else tk.apply(" ".join)


def semantic_rank(query, embedder, art, article_embeddings, top_k=50):
    q = embedder.encode(str(query), convert_to_tensor=True).detach().cpu().unsqueeze(0)  # shape (1, dim)
    results = []

    # prepare id lists
    art_ids = art["id"].tolist()
    emb_keys = set(article_embeddings.keys())

    # existing embeddings
    known_ids = [
        id
        for id in art_ids
        if id in emb_keys
    ]

    print("\tCalculating scores for known articles...")
    if len(known_ids) > 0:
        stacked = torch.stack([article_embeddings[i].detach().cpu() for i in known_ids], dim=0)  # (N, dim)
        sims_existing = util.cos_sim(q, stacked)[0]  # tensor length N
        for i, aid in enumerate(known_ids):
            results.append((aid, float(sims_existing[i].item()), False))

    # new articles: join text columns, encode, compute sims
    # build texts by joining the specified columns (simple join, no extra checks)
    new_mask = ~art["id"].isin(known_ids)
    new_art = art[new_mask]

    print("\tCalculating scores for new articles...")
    if len(new_art) > 0:
        texts = join_strcols(art, SEMANTIC_COLS).tolist()
        ids_new = art["id"].tolist()
        new_embs = embedder.encode(texts, convert_to_tensor=True).detach().cpu()  # (M, dim)
        sims_new = util.cos_sim(q, new_embs)[0]  # tensor length M
        for i, aid in enumerate(ids_new):
            results.append((aid, float(sims_new[i].item()), True))

    # sort by score desc and return top_k
    ranked = sorted(results, key=lambda x: -x[SCORE_IX])[:top_k]
    return ranked


def recommend(query, model, embedder, article_embeddings, aid_map, item_features, top_k=50):
    # Get the articles which are going to be ranked
    print("Creating semantic recommendations...")
    candidates = semantic_rank(query, embedder, article_embeddings, top_k)
    article_ids = [aid for aid, _ in candidates]

    print("Creating presonal recommendations...")
    aids = [aid_map[aid] for aid in article_ids]
    scores = model.predict(UID_PRED, aids, item_features=item_features)

    # Combine semantic + LightFM scores (weighted)
    result = [(aid, sem_score, model_score)
              for (aid, sem_score), model_score in zip(candidates, scores)]

    # Rerank the recommendations prioritizing the model
    print("Ranking results...")
    ranked = sorted(result, key=lambda x: -x[MODEL_SCORE_IX])
    return ranked


def main():
    # Initiate communiaction with the database
    cred = credentials.Certificate("../data/firebase-cred.json")
    firebase_admin.initialize_app(cred)

    config.db = firestore.client()
    config.answer_ref = config.db.collection("respuestas")
    coll_ref = config.db.collection("busqueda")

    # Get the model data
    data = joblib.load("../data/model-data.pkl")

    config.model = data["model"]
    config.item_features = data["item_features"]
    ds = data["dataset"]
    config.embedder = data["embedder"]
    config.article_embeddings = data["article_embeddings"]

    #config.articles = pd.read_csv("../data/noticias_econojournal_completo.csv")
    config.articles = pd.read_csv("../data/new_articles.csv")
    config.articles = config.articles.rename(columns={
        "ID": "id",
        "Título": "title",
        "Descripción": "description",
        "Cuerpo": "body"
    })

    _, _, config.aid_map, _ = ds.mapping()
    semantic_rank("algo", config.embedder, config.articles, config.article_embeddings)


    '''
    coll_ref.on_snapshot(on_snapshot)

    print("Listening...")
    while True:
        pass
    '''


if __name__ == "__main__":
    main()
