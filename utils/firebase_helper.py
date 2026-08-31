import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import datetime
import os
import streamlit as st


# ============================================================
# INIT FIREBASE
# ============================================================

def init_firebase():
    """
    Initialize Firebase Admin SDK once and return
    a Firestore client.

    Supports:
    1. Streamlit Cloud using st.secrets["firebase"]
    2. Local development using FIREBASE_KEY_PATH
       or firebase_key.json
    """

    if not firebase_admin._apps:

        try:
            # ------------------------------------------------
            # Streamlit Cloud
            # ------------------------------------------------
            firebase_credentials = dict(st.secrets["firebase"])

            # Convert escaped \n characters into actual
            # newlines inside the private key.
            if "private_key" in firebase_credentials:
                firebase_credentials["private_key"] = (
                    firebase_credentials["private_key"]
                    .replace("\\n", "\n")
                )

            cred = credentials.Certificate(firebase_credentials)

        except Exception:
            # ------------------------------------------------
            # Local development fallback
            # ------------------------------------------------
            firebase_path = os.getenv(
                "FIREBASE_KEY_PATH",
                "firebase_key.json"
            )

            cred = credentials.Certificate(firebase_path)

        firebase_admin.initialize_app(cred)

    return firestore.client()


# ============================================================
# UPLOAD CSV TO FIRESTORE
# ============================================================

def upload_dataframe_to_firestore(
    df,
    collection="transports"
):
    """
    Replace the existing Firestore transport collection
    with the supplied dataframe.
    """

    db = init_firebase()
    col_ref = db.collection(collection)

    # --------------------------------------------------------
    # Delete existing documents
    # --------------------------------------------------------

    existing = col_ref.get()

    for doc in existing:
        doc.reference.delete()

    # --------------------------------------------------------
    # Upload dataframe using Firestore batches
    # --------------------------------------------------------

    batch = db.batch()
    count = 0

    for _, row in df.iterrows():

        doc_ref = col_ref.document(
            str(row["transport_id"])
        )

        row_dict = {}

        for k, v in row.to_dict().items():

            if hasattr(v, "item"):
                row_dict[k] = v.item()

            else:
                row_dict[k] = v

        batch.set(doc_ref, row_dict)

        count += 1

        # Firestore batch safety
        if count % 400 == 0:

            batch.commit()
            batch = db.batch()

    # Commit remaining documents
    batch.commit()

    return count


# ============================================================
# FETCH TRANSPORT DATA FROM FIRESTORE
# ============================================================

def fetch_from_firestore(
    collection="transports"
):
    """
    Fetch transport records from Firestore and restore
    categorical values if they were stored as numeric codes.
    """

    db = init_firebase()

    docs = db.collection(collection).get()

    if not docs:
        return pd.DataFrame()

    df = pd.DataFrame(
        [
            doc.to_dict()
            for doc in docs
        ]
    )

    # --------------------------------------------------------
    # Restore module values
    # --------------------------------------------------------

    module_mapping = {
        0: "FI",
        1: "MM",
        2: "SD",
        3: "HR",

        "0": "FI",
        "1": "MM",
        "2": "SD",
        "3": "HR"
    }

    # --------------------------------------------------------
    # Restore transport stage values
    # --------------------------------------------------------

    stage_mapping = {
        0: "Development",
        1: "Quality",
        2: "Production",

        "0": "Development",
        "1": "Quality",
        "2": "Production"
    }

    # --------------------------------------------------------
    # Restore change request status values
    # --------------------------------------------------------

    status_mapping = {
        0: "Approved",
        1: "Pending",
        2: "Rejected",

        "0": "Approved",
        "1": "Pending",
        "2": "Rejected"
    }

    # --------------------------------------------------------
    # Apply mappings
    # --------------------------------------------------------

    if "module" in df.columns:

        df["module"] = df["module"].replace(
            module_mapping
        )

    if "transport_stage" in df.columns:

        df["transport_stage"] = df[
            "transport_stage"
        ].replace(
            stage_mapping
        )

    if "change_request_status" in df.columns:

        df["change_request_status"] = df[
            "change_request_status"
        ].replace(
            status_mapping
        )

    return df


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    data: dict,
    collection="predictions"
):
    """
    Save an ML prediction to Firestore.
    """

    db = init_firebase()

    clean = {}

    for k, v in data.items():

        if hasattr(v, "item"):
            clean[k] = v.item()

        else:
            clean[k] = v

    db.collection(collection).add(clean)


# ============================================================
# FETCH PREDICTIONS
# ============================================================

def fetch_predictions(
    collection="predictions"
):
    """
    Fetch prediction history ordered by timestamp.
    """

    db = init_firebase()

    docs = (
        db.collection(collection)
        .order_by(
            "timestamp",
            direction=firestore.Query.DESCENDING
        )
        .get()
    )

    if not docs:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            doc.to_dict()
            for doc in docs
        ]
    )


# ============================================================
# SAVE AI INSIGHT
# ============================================================

def save_ai_insight(
    transport_id,
    insight,
    collection="ai_insights"
):
    """
    Save an AI-generated insight for a transport.
    """

    db = init_firebase()

    db.collection(collection).document(
        str(transport_id)
    ).set(
        {
            "transport_id": str(transport_id),
            "insight": insight,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )


# ============================================================
# SAVE RELEASE OVERRIDE
# ============================================================

def save_release_override(
    transport_id,
    ai_recommendation,
    ai_risk_score,
    final_decision,
    override_reason,
    user_email
):
    """
    Save a human override of an AI release recommendation.

    Stored in:
        override_history

    Captures:
        - Transport ID
        - AI recommendation
        - AI risk score
        - Final human decision
        - Override action
        - Override reason
        - User email
        - Timestamp
    """

    db = init_firebase()

    db.collection(
        "override_history"
    ).add(
        {
            "transport_id": str(
                transport_id
            ),

            "ai_recommendation": str(
                ai_recommendation
            ),

            "ai_risk_score": float(
                ai_risk_score
            ),

            "final_decision": str(
                final_decision
            ),

            "action": "HUMAN_OVERRIDE",

            "override_reason": str(
                override_reason
            ),

            "user_email": str(
                user_email
            ),

            "timestamp": datetime.datetime.now().isoformat()
        }
    )


# ============================================================
# FETCH AI INSIGHTS
# ============================================================

def fetch_ai_insights(
    collection="ai_insights"
):
    """
    Fetch all stored AI insights.
    """

    db = init_firebase()

    docs = db.collection(collection).get()

    if not docs:
        return {}

    return {
        doc.id: doc.to_dict()
        for doc in docs
    }


# ============================================================
# SAVE CHAT HISTORY
# ============================================================

def save_chat_history(
    session_id,
    messages,
    collection="chat_history"
):
    """
    Save chat messages for a session.
    """

    db = init_firebase()

    db.collection(collection).document(
        session_id
    ).set(
        {
            "session_id": session_id,
            "messages": messages,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )


# ============================================================
# FETCH CHAT HISTORY
# ============================================================

def fetch_chat_history(
    session_id,
    collection="chat_history"
):
    """
    Fetch chat history for a session.
    """

    db = init_firebase()

    doc = (
        db.collection(collection)
        .document(session_id)
        .get()
    )

    if doc.exists:

        return doc.to_dict().get(
            "messages",
            []
        )

    return []


# ============================================================
# DELETE ALL DOCUMENTS IN COLLECTION
# ============================================================

def clear_collection(
    collection
):
    """
    Delete all documents from a Firestore collection.

    Returns the number of deleted documents.
    """

    db = init_firebase()

    docs = db.collection(collection).get()

    for doc in docs:

        doc.reference.delete()

    return len(docs)