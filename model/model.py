import re
import torch
import joblib
import numpy as np
import pandas as pd
import firebase_admin

from scipy.sparse import csr_matrix
from firebase_admin import credentials
from firebase_admin import firestore
from sentence_transformers import util

import config
from client import *

UID_PRED = 0

SCORE_IX = 1
MODEL_SCORE_IX = 2

TOP_K_WORDS = 50

TFIDF_MATRIX_ROW_IX = 0

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


def tfidf_filter(matrix, top_k=20):
    rows, cols, data = [], [], []

    for i in range(matrix.shape[TFIDF_MATRIX_ROW_IX]):
        row = matrix.getrow(i)
        # Skip if it doesn't have stored values
        if row.nnz == 0: continue

        # Indices of the top-k
        top_indices = np.argsort(row.data)[-top_k:]
        chosen_cols = row.indices[top_indices]
        chosen_vals = row.data[top_indices]

        # Build the sparse matrix
        rows.extend([i] * len(chosen_cols))
        cols.extend(chosen_cols)
        data.extend(chosen_vals)

    return csr_matrix((data, (rows, cols)), shape=matrix.shape)


def build_item_features_data(items, terms, tfidf_topk):
    item_feature_list = []

    for doc_idx, aid in enumerate(items):
        feat_dict = {}
        row = tfidf_topk.getrow(doc_idx)

        # Term -> score
        for term_idx, score in zip(row.indices, row.data):
            feat_name = f"tf:{terms[term_idx]}"
            feat_dict[feat_name] = float(score)

        item_feature_list.append((str(aid), feat_dict))

    return item_feature_list


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


def model_recommend_new(new_articles_df,
                        join_strcols,
                        tfidf,
                        tfidf_filter,
                        TOP_K_WORDS,
                        terms,
                        build_item_features_data,
                        ds,
                        model,
                        item_features,
                        uid_pred):
    """
    Build item_features rows for all new_articles_df and compute model scores.

    Returns: dict {article_id: score}
    """
    if new_articles_df is None or len(new_articles_df) == 0:
        return {}

    # 1) Prepare text series and transform to TF-IDF
    texts = join_strcols(new_articles_df, SEMANTIC_COLS).tolist()
    new_ids = new_articles_df["id"].astype(str).tolist()

    new_tfidf = tfidf.transform(texts)
    new_tfidf_topk = tfidf_filter(new_tfidf, TOP_K_WORDS)

    # 2) Build feature dicts for each new item
    new_item_data = build_item_features_data(new_ids, terms, new_tfidf_topk)
    # new_item_data: list of (id, feat_dict)

    # 3) Map feature names -> indices
    _, _, _, feat_map = ds.mapping()
    _, n_features = item_features.shape

    # 4) Construct sparse matrix with one row per new item
    data = []
    row_ind = []
    col_ind = []

    for row_idx, (aid, feat_dict) in enumerate(new_item_data):
        for feat_name, weight in feat_dict.items():
            if feat_name in feat_map:
                col = feat_map[feat_name]
                data.append(float(weight))
                row_ind.append(row_idx)
                col_ind.append(col)
    if len(data) == 0:
        new_feat_matrix = csr_matrix((len(new_item_data), n_features), dtype=np.float32)
    else:
        new_feat_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(new_item_data), n_features), dtype=np.float32)

    # 5) Get item representations for the batch
    new_item_biases, new_item_embs = model.get_item_representations(new_feat_matrix)
    # 6) Get user representations once
    user_biases, user_embs = model.get_user_representations()
    user_vec = np.array(user_embs[uid_pred])
    user_bias_val = float(user_biases[uid_pred])

    # 7) Compute scores for each new item
    scores = {}
    for i, (aid, _) in enumerate(new_item_data):
        item_emb = np.array(new_item_embs[i])
        item_bias = float(new_item_biases[i])
        score = float(np.dot(user_vec, item_emb) + user_bias_val + item_bias)
        scores[aid] = score

    return scores


def recommend(query, embedder, art, article_embeddings, aid_map, item_features, top_k=50):
    """
    Hybrid recommender that uses semantic_rank_new to get candidates (id, sem_score, is_new),
    then scores known items with model.predict and unknown items with model_recommend_new.
    Returns list of (aid, sem_score, model_score) sorted by model_score desc.
    """

    # 1) semantic candidates (expects semantic_rank_new returns (id, sem_score, is_new))
    print("Creating semantic recommendations...")
    candidates = semantic_rank(query, embedder, art, article_embeddings, top_k)
    article_ids = [aid for aid, _, _ in candidates]

    # 2) split known / unknown
    known_ids = [aid for aid, _, is_new in candidates if not is_new]
    unknown_ids = [aid for aid, _, is_new in candidates if is_new]

    model_scores_map = {}

    # 3) predict for known ids (batched)
    print("Calculating personal scores...")
    print("\tCalculating scores for known articles...")
    if known_ids:
        model_item_ids = [aid_map[aid] for aid in known_ids]
        preds = config.model.predict(UID_PRED, model_item_ids, item_features=item_features)
        for aid, p in zip(known_ids, preds):
            model_scores_map[aid] = float(p)

    # 4) predict for unknown ids using model_recommend_new (batch)
    print("\tCalculating scores for new articles...")
    if unknown_ids:
        # subset art DataFrame to only unknown_ids preserving order
        subset = art[art["id"].isin(unknown_ids)].copy()
        subset["__order__"] = subset["id"].apply(lambda x: unknown_ids.index(x))
        subset = subset.sort_values("__order__").drop(columns="__order__").reset_index(drop=True)

        new_scores = model_recommend_new(
            new_articles_df=subset,
            join_strcols=join_strcols,        # your helper
            tfidf=config.tfidf,
            tfidf_filter=tfidf_filter,
            TOP_K_WORDS=TOP_K_WORDS,
            terms=config.terms,
            build_item_features_data=build_item_features_data,
            ds=config.ds,
            model=config.model,
            item_features=item_features,
            uid_pred=UID_PRED
        )
        # new_scores is dict {id: score}
        model_scores_map.update(new_scores)

    # 5) combine semantic + model scores
    sem_map = {aid: sem for aid, sem, _ in candidates}
    combined = []
    for aid in article_ids:
        sem_score = sem_map.get(aid, 0.0)
        model_score = model_scores_map.get(aid, 0.0)
        combined.append((aid, sem_score, model_score))

    # 6) rerank by model score and return
    ranked = sorted(combined, key=lambda x: -x[2])
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
    config.ds = data["dataset"]
    config.embedder = data["embedder"]
    config.article_embeddings = data["article_embeddings"]
    config.tfidf = data["tfidf"]
    config.terms = data["terms"]

    #config.articles = pd.read_csv("../data/noticias_econojournal_completo.csv")
    config.articles = pd.read_csv("../data/new_articles.csv")
    config.articles = config.articles.rename(columns={
        "ID": "id",
        "Título": "title",
        "Descripción": "description",
        "Cuerpo": "body"
    })

    _, _, config.aid_map, _ = config.ds.mapping()
    #semantic_rank("algo", config.embedder, config.articles, config.article_embeddings)
    recommend("algo", config.embedder, config.articles, config.article_embeddings, config.aid_map, config.item_features)


    '''
    coll_ref.on_snapshot(on_snapshot)

    print("Listening...")
    while True:
        pass
    '''


if __name__ == "__main__":
    main()
