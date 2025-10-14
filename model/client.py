import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore


def on_snapshot(col_snapshot, changes, read_time):
    global is_init_loaded

    if is_init_loaded == False:
        print("Loaded first docs...")
        is_init_loaded = True

        return

    
    print(f"Snapshot received at {read_time}")
    for change in changes:
        if change.type.name == "ADDED":
            print(f"New document: {change.document.id} => {change.document.to_dict()}")


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


cred = credentials.Certificate("../firebase-cred.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
coll_ref = db.collection("busquedas")

is_init_loaded = False
watch = coll_ref.on_snapshot(on_snapshot)

print("Listening...")
while True:
    pass
