import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import config
from model import *


def on_snapshot(col_snapshot, changes, read_time):
    if not config.is_init_loaded:
        print("Loaded first docs...")
        config.is_init_loaded = True

        return

    #results = recommend("dólar",
    #                    config.model,
    #                    config.embedder,
    #                    config.article_embeddings,
    #                    config.aid_map,
    #                    config.item_features, 3)
    #for aid, _, _ in results:
    #    temp = config.articles.loc[config.articles["id"] == aid]["title"].squeeze()
    #    print(f"{aid} | {temp}")

    print(f"Snapshot received at {read_time}")
    for change in changes:
        if change.type.name == "ADDED":
            doc = change.document.to_dict()
            msg = doc.get("palabra")
            print(msg)

            print(f"Mensaje: {msg}")
            results = recommend(msg,
                                config.embedder,
                                config.articles,
                                config.article_embeddings,
                                config.aid_map,
                                config.item_features)
            doc_data = {
                "message": msg,
                "ids": [],
                "titles": [],
                "descriptions": [],
                "bodies": [],
                "imgs": [],
                "urls": [],
                "keywords": [],
                "date": []
            }

            print("Creating formatted output...")
            for aid, _, _ in results:
                temp = config.articles.loc[config.articles["id"] == aid]

                doc_data["ids"].append(aid)
                doc_data["titles"].append(temp["title"].squeeze())
                doc_data["descriptions"].append(temp["description"].squeeze())
                doc_data["bodies"].append(temp["body"].squeeze())
                doc_data["imgs"].append(temp["Imagen"].squeeze())
                doc_data["urls"].append(temp["URL"].squeeze())
                doc_data["keywords"].append(temp["PalabraClave"].squeeze())
                doc_data["date"].append(temp["Fecha"].squeeze())

                temp = temp["title"].squeeze()
                print(f"{aid} | {temp}")

            config.answer_ref.add(doc_data)



def get_collection(coll_name, db):
    # Object that has the collection data
    docs = (db.collection(coll_name).stream())

    doc_list = []
    # Add the id and the data of every message into the list
    for doc in docs:
        doc_data = doc.to_dict()
        doc_data["id"] = doc.id
        doc_data["data"] = doc._data

        doc_list.append(doc_data)

    return doc_list
