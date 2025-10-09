import  firebase_admin
from    firebase_admin import credentials
from    firebase_admin import firestore

cred = credentials.Certificate("./firebase-cred.json")
firebase_admin.initialize_app(cred)

db      = firestore.client()
data    = {
    "mensaje": "msg",
    "usuario": "Python"
}

doc_ref = db.collection("mensajes").document()
doc_ref.set(data)
