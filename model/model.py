import re
import joblib
import pandas as pd

from sentence_transformers import util


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
    data = joblib.load("model-data.pkl")

    model = data["model"]
    item_features = data["item_features"]
    ds = data["dataset"]
    embedder = data["embedder"]
    article_embeddings = data["article_embeddings"]

    articles = pd.read_csv("/content/drive/MyDrive/Geinsa/data/noticias_econojournal_completo.csv")
    articles = articles.rename(columns={
        "ID": "id",
        "Título": "title",
        "Descripción": "description",
        "Cuerpo": "body"
    })

    _, _, aid_map, _ = ds.mapping()

    results = recommend("dólar", model, embedder, article_embeddings, aid_map, item_features, 3)
    for aid, _, _ in results:
        temp = articles.loc[articles["id"] == aid]["title"].squeeze()
        print(f"{aid} | {temp}")


if __name__ == "__main__":
    main()
