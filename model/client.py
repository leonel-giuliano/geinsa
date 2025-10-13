import  firebase_admin
from    firebase_admin                          import credentials
from    firebase_admin                          import firestore
from    google.cloud.firestore_v1.base_query    import FieldFilter, Or


def get_collection(coll_name):
    # Object that has the collection data
    docs = (db.collection(coll_name).stream())

    doc_list = []
    # Add the id and the data of every message into the list
    for doc in docs:
        doc_data            = doc.to_dict()
        doc_data["id"]      = doc.id
        doc_data["data"]    = doc._data

        doc_list.append(doc_data)

    return doc_list


cred = credentials.Certificate("./firebase-cred.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

collection = get_collection("mensajes")
for doc in collection:
    print(f"Id: {doc["id"]}")
    print(f"Data: {doc["data"]}")
