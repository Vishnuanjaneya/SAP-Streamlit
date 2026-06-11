import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import os
import datetime
import streamlit as st

firebase_credentials = dict(st.secrets["firebase"])
cred = credentials.Certificate(firebase_credentials)

# ---------------- INIT FIREBASE ----------------
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(
            os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
        )
        firebase_admin.initialize_app(cred)
    return firestore.client()

# ---------------- UPLOAD CSV TO FIRESTORE ----------------
def upload_dataframe_to_firestore(df, collection="transports"):
    db = init_firebase()
    col_ref = db.collection(collection)

    # Clear existing
    existing = col_ref.get()
    for doc in existing:
        doc.reference.delete()

    # Upload in batches
    batch = db.batch()
    count = 0
    for i, row in df.iterrows():
        doc_ref = col_ref.document(str(row['transport_id']))
        row_dict = {}
        for k, v in row.to_dict().items():
            # Convert numpy types to native Python
            if hasattr(v, 'item'):
                row_dict[k] = v.item()
            else:
                row_dict[k] = v
        batch.set(doc_ref, row_dict)
        count += 1
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()

    batch.commit()
    return count

# ---------------- FETCH FROM FIRESTORE ----------------
def fetch_from_firestore(collection="transports"):
    db = init_firebase()
    docs = db.collection(collection).get()
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame([doc.to_dict() for doc in docs])

    # ✅ Fix numeric encoded columns back to original names
    module_mapping = {
        0: 'FI', 1: 'MM', 2: 'SD', 3: 'HR',
        '0': 'FI', '1': 'MM', '2': 'SD', '3': 'HR'
    }
    stage_mapping = {
        0: 'Development', 1: 'Quality', 2: 'Production',
        '0': 'Development', '1': 'Quality', '2': 'Production'
    }
    status_mapping = {
        0: 'Approved', 1: 'Pending', 2: 'Rejected',
        '0': 'Approved', '1': 'Pending', '2': 'Rejected'
    }

    if 'module' in df.columns:
        df['module'] = df['module'].replace(module_mapping)
    if 'transport_stage' in df.columns:
        df['transport_stage'] = df['transport_stage'].replace(stage_mapping)
    if 'change_request_status' in df.columns:
        df['change_request_status'] = df['change_request_status'].replace(status_mapping)

    return df

# ---------------- SAVE PREDICTION ----------------
def save_prediction(data: dict, collection="predictions"):
    db = init_firebase()
    # Convert any numpy types
    clean = {}
    for k, v in data.items():
        if hasattr(v, 'item'):
            clean[k] = v.item()
        else:
            clean[k] = v
    db.collection(collection).add(clean)

# ---------------- FETCH PREDICTIONS ----------------
def fetch_predictions(collection="predictions"):
    db = init_firebase()
    docs = db.collection(collection).order_by(
        "timestamp", direction=firestore.Query.DESCENDING
    ).get()
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame([doc.to_dict() for doc in docs])

# ---------------- SAVE AI INSIGHT ----------------
def save_ai_insight(transport_id, insight, collection="ai_insights"):
    db = init_firebase()
    db.collection(collection).document(str(transport_id)).set({
        "transport_id": str(transport_id),
        "insight": insight,
        "timestamp": datetime.datetime.now().isoformat()
    })

# ---------------- FETCH AI INSIGHTS ----------------
def fetch_ai_insights(collection="ai_insights"):
    db = init_firebase()
    docs = db.collection(collection).get()
    if not docs:
        return {}
    return {doc.id: doc.to_dict() for doc in docs}

# ---------------- SAVE CHAT HISTORY ----------------
def save_chat_history(session_id, messages, collection="chat_history"):
    db = init_firebase()
    db.collection(collection).document(session_id).set({
        "session_id": session_id,
        "messages": messages,
        "timestamp": datetime.datetime.now().isoformat()
    })

# ---------------- FETCH CHAT HISTORY ----------------
def fetch_chat_history(session_id, collection="chat_history"):
    db = init_firebase()
    doc = db.collection(collection).document(session_id).get()
    if doc.exists:
        return doc.to_dict().get("messages", [])
    return []

# ---------------- DELETE ALL IN COLLECTION ----------------
def clear_collection(collection):
    db = init_firebase()
    docs = db.collection(collection).get()
    for doc in docs:
        doc.reference.delete()
    return len(docs)