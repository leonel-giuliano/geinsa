import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from model import *


def on_snapshot(col_snapshot, changes, read_time):
    global is_init_loaded

    if not is_init_loaded:
        print("Loaded first docs...")
        is_init_loaded = True

        return

    results = recommend("dólar", model, embedder, article_embeddings, aid_map, item_features, 3)
    for aid, _, _ in results:
        temp = articles.loc[articles["id"] == aid]["title"].squeeze()
        print(f"{aid} | {temp}")

    #print(f"Snapshot received at {read_time}")
    #for change in changes:
    #    if change.type.name == "ADDED":
    #        print(f"New document: {change.document.id} => {change.document.to_dict()}")


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
