import streamlit as st
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime
from sklearn.ensemble import IsolationForest
import time
import json
import uuid
import re
from xml.sax.saxutils import escape

from utils.gauge import risk_gauge
from utils.ml_helper import preprocess, predict
from utils.firebase_helper import (
    upload_dataframe_to_firestore,
    fetch_from_firestore,
    save_prediction,
    fetch_predictions,
    save_ai_insight,
    fetch_ai_insights,
    save_chat_history,
    fetch_chat_history,
    clear_collection,
    save_release_override,
)
# ---------------- FIREBASE AUTHENTICATION ----------------

def firebase_auth_url(action):
    api_key = os.getenv("FIREBASE_WEB_API_KEY")

    if not api_key:
        raise ValueError("FIREBASE_WEB_API_KEY is not configured.")

    return f"https://identitytoolkit.googleapis.com/v1/accounts:{action}?key={api_key}"


def sign_up(email, password):
    url = firebase_auth_url("signUp")

    response = requests.post(
        url,
        json={
            "email": email,
            "password": password,
            "returnSecureToken": True
        },
        timeout=15
    )

    data = response.json()

    if response.ok:
        return True, data

    return False, data.get("error", {}).get(
        "message",
        "Sign-up failed."
    )


def sign_in(email, password):
    url = firebase_auth_url("signInWithPassword")

    response = requests.post(
        url,
        json={
            "email": email,
            "password": password,
            "returnSecureToken": True
        },
        timeout=15
    )

    data = response.json()

    if response.ok:
        return True, data

    return False, data.get("error", {}).get(
        "message",
        "Sign-in failed."
    )


def logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.id_token = None
    st.session_state.refresh_token = None
    st.rerun()

# ================================================================
# CONFIG
# ================================================================
load_dotenv()
st.set_page_config(page_title="SAP Risk AI", layout="wide", page_icon="🚀")

# ---------------- AUTH SESSION ----------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "id_token" not in st.session_state:
    st.session_state.id_token = None

if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None
# ================================================================
# AUTHENTICATION GATE
# ================================================================

if not st.session_state.authenticated:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:40px 0 20px 0;
        ">
            <h1>🚀 SAP Risk AI</h1>
            <p style="font-size:18px; opacity:0.75;">
                SAP Transport Risk Decision Intelligence Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    auth_tab1, auth_tab2 = st.tabs(["🔐 Sign In", "🆕 Create Account"])

    # ---------------- SIGN IN ----------------
    with auth_tab1:

        st.subheader("Welcome back")

        login_email = st.text_input(
            "Email",
            placeholder="Enter your email",
            key="login_email"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password"
        )

        if st.button(
            "🔓 Sign In",
            type="primary",
            use_container_width=True
        ):

            if not login_email or not login_password:
                st.error("Please enter both email and password.")

            else:
                with st.spinner("Signing in..."):

                    success, result = sign_in(
                        login_email,
                        login_password
                    )

                if success:

                    st.session_state.authenticated = True
                    st.session_state.user_email = result.get(
                        "email",
                        login_email
                    )
                    st.session_state.id_token = result.get(
                        "idToken"
                    )
                    st.session_state.refresh_token = result.get(
                        "refreshToken"
                    )

                    st.success("Login successful!")
                    st.rerun()

                else:

                    error_code = result

                    error_messages = {
                        "EMAIL_NOT_FOUND":
                            "No account exists with this email.",
                        "INVALID_PASSWORD":
                            "Incorrect password.",
                        "INVALID_LOGIN_CREDENTIALS":
                            "Invalid email or password.",
                        "USER_DISABLED":
                            "This account has been disabled.",
                        "TOO_MANY_ATTEMPTS_TRY_LATER":
                            "Too many attempts. Please try again later.",
                    }

                    st.error(
                        error_messages.get(
                            error_code,
                            f"Sign-in failed: {error_code}"
                        )
                    )

    # ---------------- SIGN UP ----------------
    with auth_tab2:

        st.subheader("Create your account")

        signup_email = st.text_input(
            "Email",
            placeholder="Enter your email",
            key="signup_email"
        )

        signup_password = st.text_input(
            "Password",
            type="password",
            placeholder="Create a password",
            key="signup_password"
        )

        signup_confirm = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Re-enter your password",
            key="signup_confirm"
        )

        if st.button(
            "🆕 Create Account",
            type="primary",
            use_container_width=True
        ):

            if not signup_email or not signup_password:
                st.error(
                    "Please enter your email and password."
                )

            elif signup_password != signup_confirm:
                st.error("Passwords do not match.")

            elif len(signup_password) < 6:
                st.error(
                    "Password must contain at least 6 characters."
                )

            else:

                with st.spinner("Creating your account..."):

                    success, result = sign_up(
                        signup_email,
                        signup_password
                    )

                if success:

                    st.success(
                        "Account created successfully! "
                        "Please go to Sign In and log in."
                    )

                else:

                    error_code = result

                    error_messages = {
                        "EMAIL_EXISTS":
                            "An account with this email already exists.",
                        "INVALID_EMAIL":
                            "Please enter a valid email address.",
                        "WEAK_PASSWORD":
                            "Password is too weak.",
                    }

                    st.error(
                        error_messages.get(
                            error_code,
                            f"Sign-up failed: {error_code}"
                        )
                    )

    # Don't load the SAP application until authenticated
    st.stop()
# ============================================================
# USER PROFILE / LOGOUT BAR
# ============================================================
# Top-right authenticated user area.
# Email/password Firebase authentication does not provide a Google
# profile photo, so the avatar is represented by the first letter.
# ============================================================

profile_spacer, profile_user, profile_action = st.columns([7, 2.5, 1.1])

with profile_user:
    user_email = st.session_state.get("user_email") or "User"
    avatar_letter = user_email[0].upper() if user_email else "U"

    avatar_col, email_col = st.columns([0.65, 2.35])

    with avatar_col:
        st.markdown(
            f"""
            <div style="
                width:42px;
                height:42px;
                border-radius:50%;
                background:linear-gradient(135deg,#4285F4,#34A853);
                display:flex;
                align-items:center;
                justify-content:center;
                color:white;
                font-weight:700;
                font-size:18px;
                margin-top:2px;
                box-shadow:0 2px 8px rgba(0,0,0,0.25);
            ">{avatar_letter}</div>
            """,
            unsafe_allow_html=True
        )

    with email_col:
        st.write(user_email)

with profile_action:
    if st.button(
        "🚪 Logout",
        key="top_logout",
        width="stretch",
        help="Sign out of SAP Risk AI"
    ):
        logout()

st.divider()


# Session ID for chat persistence
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

# ================================================================
# LANDING PAGE HERO
# ================================================================
st.markdown("""
<div style='background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
     padding: 40px 30px; border-radius: 20px; text-align: center;
     margin-bottom: 25px; border: 1px solid #764ba2;
     box-shadow: 0 0 40px rgba(118,75,162,0.4)'>
    <div style='font-size:3em; margin-bottom:10px'>🚀</div>
    <h1 style='color: white; font-size: 2.4em; margin:0; letter-spacing:1px'>
        SAP Transport Risk Intelligence
    </h1>
    <p style='color: #b39ddb; font-size: 1.15em; margin-top:10px'>
        ML Prediction &nbsp;•&nbsp; LLM Insights &nbsp;•&nbsp;
        Firebase Cloud &nbsp;•&nbsp; Anomaly Detection &nbsp;•&nbsp; Real-Time Monitoring
    </p>
    <div style='margin-top:15px; display:flex; justify-content:center; gap:12px; flex-wrap:wrap'>
        <span style='background:#4caf50;color:white;padding:4px 14px;
              border-radius:20px;font-size:0.85em'>✅ AI-Powered</span>
        <span style='background:#2196f3;color:white;padding:4px 14px;
              border-radius:20px;font-size:0.85em'>📊 Enterprise-Grade</span>
        <span style='background:#9c27b0;color:white;padding:4px 14px;
              border-radius:20px;font-size:0.85em'>🔥 Firebase Cloud</span>
        <span style='background:#ff5722;color:white;padding:4px 14px;
              border-radius:20px;font-size:0.85em'>⚡ Real-Time</span>
        <span style='background:#607d8b;color:white;padding:4px 14px;
              border-radius:20px;font-size:0.85em'>🔍 Anomaly Detection</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# GROQ SETUP
# ================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client  = Groq(api_key=GROQ_API_KEY)

# ================================================================
# LOAD MODEL
# ================================================================
try:
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"⚠ Model loading failed: {type(e).__name__}: {e}")
    st.stop()

# ================================================================
# HELPERS
# ================================================================
def save_chart(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def ask_ai(prompt: str) -> str:
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {e}"

def ask_ai_with_history(messages: list) -> str:
    try:
        res = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ AI Error: {e}"

def get_confidence(df_proc):
    try:
        proba = model.predict_proba(df_proc)
        return (proba.max(axis=1) * 100).round(2), model.classes_, proba
    except:
        return None, None, None


def show_model_performance():
    """Display trained model performance metrics and confusion matrix."""

    st.markdown("### 📈 Model Performance")

    metrics_path = "model/model_metrics.json"
    matrix_path = "model/confusion_matrix.png"

    # Load and display model metrics
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            col1, col2, col3, col4 = st.columns(4)

            accuracy = metrics.get("accuracy")
            precision = metrics.get("precision")
            recall = metrics.get("recall")
            f1 = metrics.get("f1_score", metrics.get("f1"))

            if accuracy is not None:
                col1.metric(
                    "🎯 Accuracy",
                    f"{float(accuracy) * 100:.2f}%"
                    if float(accuracy) <= 1
                    else f"{float(accuracy):.2f}%"
                )

            if precision is not None:
                col2.metric(
                    "🔹 Precision",
                    f"{float(precision) * 100:.2f}%"
                    if float(precision) <= 1
                    else f"{float(precision):.2f}%"
                )

            if recall is not None:
                col3.metric(
                    "🔹 Recall",
                    f"{float(recall) * 100:.2f}%"
                    if float(recall) <= 1
                    else f"{float(recall):.2f}%"
                )

            if f1 is not None:
                col4.metric(
                    "⭐ F1 Score",
                    f"{float(f1) * 100:.2f}%"
                    if float(f1) <= 1
                    else f"{float(f1):.2f}%"
                )

            with st.expander("📋 View Complete Model Metrics"):
                st.json(metrics)

        except Exception as e:
            st.warning(f"⚠ Could not load model metrics: {e}")

    else:
        st.info("ℹ️ Model metrics file not found.")

    # Display confusion matrix
    if os.path.exists(matrix_path):
        st.markdown("### 🔲 Confusion Matrix")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            st.image(
                matrix_path,
                caption="CatBoost Model Confusion Matrix",
                use_container_width=True
            )
    else:
        st.info("ℹ️ Confusion matrix image not found.")

def clean_for_pdf(text: str) -> str:
    """Sanitize AI-generated text so reportlab's Paragraph parser doesn't choke
    on markdown tables, unescaped &, <, >, or malformed HTML-like tags."""
    if not text:
        return ""
    # Strip markdown table pipes/separators - convert to plain readable text
    text = re.sub(r'\|', ' - ', text)
    text = re.sub(r'-{3,}', '', text)
    # Escape XML-special chars first so stray & < > don't break the parser
    text = escape(text)
    # Re-allow only the tags reportlab actually supports, correctly closed
    text = re.sub(r'&lt;br\s*/?&gt;', '<br/>', text, flags=re.IGNORECASE)
    text = re.sub(r'&lt;b&gt;', '<b>', text, flags=re.IGNORECASE)
    text = re.sub(r'&lt;/b&gt;', '</b>', text, flags=re.IGNORECASE)
    text = re.sub(r'&lt;i&gt;', '<i>', text, flags=re.IGNORECASE)
    text = re.sub(r'&lt;/i&gt;', '</i>', text, flags=re.IGNORECASE)
    return text.strip()

def generate_pdf(filtered_df, ai_summary, date_range=None):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story  = []

    story.append(Paragraph("SAP Transport Risk Intelligence Report", styles['Title']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['Normal']
    ))
    if date_range:
        story.append(Paragraph(f"Date Range: {date_range}", styles['Normal']))
    story.append(Spacer(1, 20))

    high  = (filtered_df['Predicted Risk']=="HIGH").sum()
    med   = (filtered_df['Predicted Risk']=="MEDIUM").sum()
    low   = (filtered_df['Predicted Risk']=="LOW").sum()
    total = len(filtered_df)

    # Summary Table
    story.append(Paragraph("Risk Summary", styles['Heading1']))
    story.append(Spacer(1, 8))
    s_data = [
        ['Metric','Value'],
        ['Total Transports', str(total)],
        ['HIGH Risk',        str(high)],
        ['MEDIUM Risk',      str(med)],
        ['LOW Risk',         str(low)],
        ['High Risk %',      f"{round(high/total*100,2)}%" if total else "0%"],
    ]
    st_ = Table(s_data, colWidths=[3*inch,3*inch])
    st_.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#302b63')),
        ('TEXTCOLOR', (0,0),(-1,0),colors.white),
        ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
        ('ALIGN',     (0,0),(-1,-1),'CENTER'),
        ('GRID',      (0,0),(-1,-1),1,colors.black),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.HexColor('#ecf0f1'),colors.white]),
    ]))
    story.append(st_)
    story.append(Spacer(1,20))

    # Module Table
    story.append(Paragraph("Module-wise Risk Breakdown", styles['Heading1']))
    story.append(Spacer(1,8))
    m_data = [['Module','HIGH','MEDIUM','LOW','Total']]
    for mod in filtered_df['module'].unique():
        mdf = filtered_df[filtered_df['module']==mod]
        m_data.append([
            mod,
            str((mdf['Predicted Risk']=="HIGH").sum()),
            str((mdf['Predicted Risk']=="MEDIUM").sum()),
            str((mdf['Predicted Risk']=="LOW").sum()),
            str(len(mdf))
        ])
    mt = Table(m_data, colWidths=[1.5*inch,1.2*inch,1.2*inch,1.2*inch,1.2*inch])
    mt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e74c3c')),
        ('TEXTCOLOR', (0,0),(-1,0),colors.white),
        ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
        ('ALIGN',     (0,0),(-1,-1),'CENTER'),
        ('GRID',      (0,0),(-1,-1),1,colors.black),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),
         [colors.HexColor('#fadbd8'),colors.white]),
    ]))
    story.append(mt)
    story.append(Spacer(1,20))

    # Critical Transports
    critical = filtered_df[
        (filtered_df['transport_stage']=="Production") &
        (filtered_df['Predicted Risk']=="HIGH")
    ]
    if not critical.empty:
        story.append(Paragraph(
            "Critical Transports (HIGH Risk in Production)", styles['Heading1']
        ))
        story.append(Spacer(1,8))
        c_data = [['Transport ID','Module','Stage','Risk']]
        for _, row in critical.head(20).iterrows():
            c_data.append([
                str(row['transport_id']), str(row['module']),
                str(row['transport_stage']), str(row['Predicted Risk'])
            ])
        ct = Table(c_data, colWidths=[2*inch,1.5*inch,1.5*inch,1.5*inch])
        ct.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#c0392b')),
            ('TEXTCOLOR', (0,0),(-1,0),colors.white),
            ('FONTNAME',  (0,0),(-1,0),'Helvetica-Bold'),
            ('ALIGN',     (0,0),(-1,-1),'CENTER'),
            ('GRID',      (0,0),(-1,-1),1,colors.black),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),
             [colors.HexColor('#fadbd8'),colors.white]),
        ]))
        story.append(ct)
        story.append(Spacer(1,20))

    # AI Summary
    story.append(Paragraph("AI-Generated Risk Analysis", styles['Heading1']))
    story.append(Spacer(1,8))
    clean = ai_summary.replace('*','').replace('#','')
    for line in clean.split('\n'):
        if line.strip():
            story.append(Paragraph(clean_for_pdf(line.strip()), styles['Normal']))
            story.append(Spacer(1, 5))
    doc.build(story)
    buf.seek(0)
    return buf

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.markdown("## 📂 Data Source")

data_source = st.sidebar.radio(
    "Choose Source",
    ["📁 Upload CSV", "🔥 Load from Firebase"]
)

df = pd.DataFrame()

if data_source == "📁 Upload CSV":
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV file(s)", type=["csv"],
        accept_multiple_files=True
    )
    if uploaded_files:
        df = pd.concat(
            [pd.read_csv(f) for f in uploaded_files],
            ignore_index=True
        )
        st.sidebar.success(f"✅ {len(df)} records loaded")

        if st.sidebar.button("🔥 Save to Firebase"):
            with st.spinner("Uploading to Firebase Firestore..."):
                try:
                    count = upload_dataframe_to_firestore(df)
                    st.sidebar.success(f"✅ {count} records saved to Firebase!")
                except Exception as e:
                    st.sidebar.error(f"Firebase Error: {e}")

elif data_source == "🔥 Load from Firebase":
    if st.sidebar.button("🔄 Fetch from Firebase"):
        with st.spinner("Fetching from Firebase..."):
            try:
                df = fetch_from_firestore()
                if df.empty:
                    st.sidebar.warning("No data found. Upload CSV first.")
                else:
                    st.sidebar.success(f"🔥 {len(df)} records fetched!")
                    st.session_state['firebase_df'] = df
            except Exception as e:
                st.sidebar.error(f"Firebase Error: {e}")

    if 'firebase_df' in st.session_state:
        df = st.session_state['firebase_df']
        st.sidebar.info(f"🔥 Using {len(df)} Firebase records")

# Live Monitor
st.sidebar.markdown("---")
st.sidebar.markdown("## ⚡ Live Monitor")
live_mode    = st.sidebar.toggle("🔴 Enable Live Monitor", value=False)
refresh_rate = 30
if live_mode:
    refresh_rate = st.sidebar.slider("Refresh every (seconds)", 10, 120, 30)
    st.sidebar.success(f"Auto-refreshing every {refresh_rate}s")

# Firebase Admin
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔥 Firebase Admin")
if st.sidebar.button("🗑 Clear Predictions History"):
    try:
        n = clear_collection("predictions")
        st.sidebar.success(f"✅ Cleared {n} predictions")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ================================================================
# MAIN APP
# ================================================================
if not df.empty:

    if live_mode:
        st.markdown("""
        <div style='background:#1a1a2e;border:2px solid #e74c3c;
             border-radius:10px;padding:10px 20px;
             display:flex;align-items:center;gap:10px;margin-bottom:15px'>
            <span style='color:#e74c3c;font-size:1.5em'>●</span>
            <span style='color:white;font-weight:bold'>LIVE MONITORING ACTIVE</span>
            <span style='color:#aaa;font-size:0.85em'>
                — Auto-refreshing every {refresh_rate}s
            </span>
        </div>
        """.replace("{refresh_rate}", str(refresh_rate)), unsafe_allow_html=True)

    st.success(f"✅ {len(df)} transports loaded")
    st.dataframe(df.head(), use_container_width=True)

    df_original  = df.copy()

    # ✅ Clean NaN values from Firebase data
    df_original = df_original.dropna(subset=['transport_id'])

    # ✅ Fix numeric module names from Firebase encoding
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
    df_original['module'] = df_original['module'].replace(module_mapping)
    df_original['transport_stage'] = df_original['transport_stage'].replace(stage_mapping)
    df_original['change_request_status'] = df_original['change_request_status'].replace(status_mapping)
    df_original['module'] = df_original['module'].fillna('Unknown')
    df_original['transport_stage'] = df_original['transport_stage'].fillna('Unknown')
    df_original['change_request_status'] = df_original['change_request_status'].fillna('Unknown')
    df_original['objects_changed'] = pd.to_numeric(df_original.get('objects_changed', 0), errors='coerce').fillna(0)
    df_original['lines_changed'] = pd.to_numeric(df_original.get('lines_changed', 0), errors='coerce').fillna(0)
    df_original['conflicts'] = pd.to_numeric(df_original.get('conflicts', 0), errors='coerce').fillna(0)
    df_original['history_failures'] = pd.to_numeric(df_original.get('history_failures', 0), errors='coerce').fillna(0)

    df_processed = preprocess(df_original.copy())

    df_original['Predicted Risk'] = predict(model, df_processed)

    conf_scores, classes, all_proba = get_confidence(df_processed)
    df_original['Confidence %'] = conf_scores if conf_scores is not None else "N/A"

    # Anomaly Detection
    numeric_cols = [c for c in
        ['objects_changed','lines_changed','conflicts','history_failures']
        if c in df_original.columns]

    if numeric_cols:
        iso = IsolationForest(contamination=0.1, random_state=42)
        iso.fit(df_original[numeric_cols])
        df_original['Anomaly'] = iso.predict(df_original[numeric_cols])
        df_original['Anomaly'] = df_original['Anomaly'].map(
            {1:"Normal", -1:"⚠ Anomaly"}
        )

    # ---- FILTERS ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔍 Filters")
    tid = st.sidebar.text_input("🔎 Transport ID")

    module_options = [x for x in df_original['module'].unique()
                      if str(x) != 'nan' and x is not None]
    module_filter  = st.sidebar.multiselect(
        "📦 Module", module_options, default=module_options
    )
    risk_options = [x for x in df_original['Predicted Risk'].unique()
                    if str(x) != 'nan' and x is not None]
    risk_filter  = st.sidebar.multiselect(
        "⚠ Risk Level", risk_options, default=risk_options
    )

    st.sidebar.markdown("## 📅 Date Filter")
    date_range_label = "All Data"
    if 'date' in df_original.columns:
        df_original['date'] = pd.to_datetime(df_original['date'], errors='coerce')
        min_date = df_original['date'].min()
        max_date = df_original['date'].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_from = st.sidebar.date_input("From", min_date)
            date_to   = st.sidebar.date_input("To",   max_date)
            date_range_label = f"{date_from} to {date_to}"
    else:
        st.sidebar.info("No 'date' column found")

    filtered_df = df_original.copy()
    if tid:
        filtered_df = filtered_df[
            filtered_df['transport_id'].astype(str).str.contains(tid)
        ]
    if module_filter:
        filtered_df = filtered_df[filtered_df['module'].isin(module_filter)]
    if risk_filter:
        filtered_df = filtered_df[filtered_df['Predicted Risk'].isin(risk_filter)]
    if 'date' in df_original.columns and pd.notna(df_original['date'].min()):
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.Timestamp(date_from)) &
            (filtered_df['date'] <= pd.Timestamp(date_to))
        ]

    # ================================================================
    # TABS
    # ================================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Dashboard",
    "🔍 Anomaly Detection",
    "📄 Data Explorer",
    "🤖 AI Insights",
    "⚡ Manual Prediction",
    "🔄 What-If Analysis",
    "📜 Prediction History",
    "💰 Business Impact & ROI",
    "🚦 Deployment Gate",
    "🚀 AI Release Commander"
])

    cmap = {"HIGH":"#ff4d4d","MEDIUM":"#f1c40f","LOW":"#2ecc71"}

    # ============================================================
    # TAB 1 — DASHBOARD
    # ============================================================
    with tab1:

        st.subheader("📊 Business Dashboard")

        high_n   = (filtered_df['Predicted Risk']=="HIGH").sum()
        med_n    = (filtered_df['Predicted Risk']=="MEDIUM").sum()
        low_n    = (filtered_df['Predicted Risk']=="LOW").sum()
        total_n  = len(filtered_df)
        critical = filtered_df[
            (filtered_df['transport_stage']=="Production") &
            (filtered_df['Predicted Risk']=="HIGH")
        ]
        anomaly_n = (filtered_df['Anomaly']=="⚠ Anomaly").sum() \
                    if 'Anomaly' in filtered_df.columns else 0

        # KPI Cards
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("📦 Total",       total_n)
        c2.metric("🔴 HIGH Risk",   high_n,
                  delta=f"{round(high_n/total_n*100,1)}%" if total_n else "0%")
        c3.metric("🟡 MEDIUM Risk", med_n)
        c4.metric("🟢 LOW Risk",    low_n)
        c5.metric("🚨 Critical",    len(critical))

        st.divider()

        # ALERT SYSTEM
        if not critical.empty:
            st.markdown(f"""
            <div style='background:linear-gradient(90deg,#c0392b,#e74c3c);
                 color:white;padding:15px 20px;border-radius:12px;
                 font-size:1.1em;font-weight:bold;margin-bottom:10px'>
                🚨 CRITICAL ALERT — {len(critical)} HIGH-RISK transports in PRODUCTION.
                Immediate action required!
            </div>""", unsafe_allow_html=True)
            with st.expander("👁 View Critical Transports"):
                st.dataframe(
                    critical[['transport_id','module','transport_stage',
                               'Predicted Risk','Confidence %']],
                    use_container_width=True
                )
        elif high_n > 0:
            st.warning(
                f"⚠ {high_n} HIGH-risk transports detected — review before deployment"
            )
        else:
            st.success("✅ All transports within acceptable risk levels")

        if anomaly_n > 0:
            st.warning(f"🔍 {anomaly_n} anomalous transports detected by AI")

        # Firebase status
        st.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid #4caf50;border-radius:8px;
             padding:8px 16px;display:inline-block;margin:8px 0'>
            <span style='color:#4caf50'>🔥 Firebase Connected</span>
            <span style='color:#aaa;font-size:0.85em'>
              &nbsp;— Data persisted to Firestore cloud database
            </span>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # Overall Gauge
        if total_n > 0:
            pct   = int((high_n/total_n)*100)
            color = "#ff4d4d" if pct>70 else "#f1c40f" if pct>40 else "#2ecc71"
            st.markdown(risk_gauge(pct, color), unsafe_allow_html=True)

        st.divider()

        # Module Gauges
        st.markdown("### 📦 Module Risk Indicators")
        modules = filtered_df['module'].unique()
        cols    = st.columns(len(modules))
        for i, mod in enumerate(modules):
            mdf  = filtered_df[filtered_df['module']==mod]
            hc   = (mdf['Predicted Risk']=="HIGH").sum()
            tot  = len(mdf)
            pmod = int((hc/tot)*100) if tot>0 else 0
            cmod = "#ff4d4d" if pmod>70 else "#f1c40f" if pmod>40 else "#2ecc71"
            with cols[i]:
                st.markdown(f"**{mod}**")
                st.markdown(risk_gauge(pmod, cmod), unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 Analytics")

        col1,col2 = st.columns(2)
        col3,col4 = st.columns(2)

        with col1:
            fig1 = px.bar(filtered_df, x="Predicted Risk",
                          title="Risk Distribution",
                          color="Predicted Risk", color_discrete_map=cmap)
            st.plotly_chart(fig1, use_container_width=True)
            mf1 = plt.figure()
            sns.countplot(x='Predicted Risk', data=filtered_df)
            st.download_button("📥 Risk Chart", save_chart(mf1), "risk.png")

        with col2:
            fig2 = px.bar(filtered_df, x="module", color="Predicted Risk",
                          title="Module vs Risk", color_discrete_map=cmap)
            st.plotly_chart(fig2, use_container_width=True)
            mf2 = plt.figure()
            sns.countplot(x='module', hue='Predicted Risk', data=filtered_df)
            st.download_button("📥 Module Chart", save_chart(mf2), "module.png")

        with col3:
            fig3 = px.histogram(filtered_df, x="module", color="Predicted Risk",
                                barmode="group", title="Clustered Risk Analysis",
                                color_discrete_map=cmap)
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = px.histogram(filtered_df, x="module", color="Predicted Risk",
                                barmode="stack", title="Stacked Risk Composition",
                                color_discrete_map=cmap)
            st.plotly_chart(fig4, use_container_width=True)

        if 'Confidence %' in filtered_df.columns and \
           filtered_df['Confidence %'].dtype != object:
            st.markdown("### 🎯 Confidence Distribution")
            fig_c = px.histogram(filtered_df, x='Confidence %',
                                 color='Predicted Risk', nbins=20,
                                 title="Model Confidence Distribution",
                                 color_discrete_map=cmap)
            st.plotly_chart(fig_c, use_container_width=True)

        st.download_button(
            "📥 Download Analytics CSV",
            data=filtered_df.to_csv(index=False),
            file_name="sap_analytics.csv", mime="text/csv"
        )

        if live_mode:
            st.info(f"🔄 Next refresh in {refresh_rate} seconds...")
            time.sleep(refresh_rate)
            st.rerun()

    # ============================================================
    # TAB 2 — ANOMALY DETECTION
    # ============================================================
    with tab2:

        st.subheader("🔍 AI Anomaly Detection")
        st.caption("Isolation Forest ML detects unusual transport patterns automatically")

        if 'Anomaly' in filtered_df.columns:
            a_cnt = (filtered_df['Anomaly']=="⚠ Anomaly").sum()
            n_cnt = (filtered_df['Anomaly']=="Normal").sum()

            c1,c2,c3 = st.columns(3)
            c1.metric("⚠ Anomalies",   a_cnt)
            c2.metric("✅ Normal",      n_cnt)
            c3.metric("Anomaly Rate",
                      f"{round(a_cnt/len(filtered_df)*100,1)}%")

            if a_cnt > 0:
                st.error(f"🚨 {a_cnt} anomalous transports detected!")
                anomalies = filtered_df[filtered_df['Anomaly']=="⚠ Anomaly"]

                st.markdown("#### ⚠ Anomalous Transports")
                st.dataframe(
                    anomalies[['transport_id','module','transport_stage',
                               'Predicted Risk','Confidence %','Anomaly']],
                    use_container_width=True
                )

                if numeric_cols and len(numeric_cols) >= 2:
                    fig_a = px.scatter(
                        filtered_df,
                        x=numeric_cols[0],
                        y=numeric_cols[1],
                        color='Anomaly',
                        symbol='Anomaly',
                        title="Anomaly Scatter Plot",
                        color_discrete_map={
                            "Normal":"#2ecc71","⚠ Anomaly":"#ff4d4d"
                        }
                    )
                    st.plotly_chart(fig_a, use_container_width=True)

                if st.button("🤖 AI Explain Anomalies"):
                    prompt = f"""
                    Anomaly detection found {a_cnt} unusual SAP transports
                    out of {len(filtered_df)}.

                    Anomalous stats:
                    {anomalies[numeric_cols].describe().to_string() if numeric_cols else 'N/A'}

                    Normal stats:
                    {filtered_df[filtered_df['Anomaly']=='Normal'][numeric_cols].describe().to_string() if numeric_cols else 'N/A'}

                    Explain:
                    1. What makes these transports anomalous
                    2. Business risk they pose
                    3. Recommended immediate actions
                    """
                    with st.spinner("AI analyzing anomalies..."):
                        st.write(ask_ai(prompt))
            else:
                st.success("✅ No anomalies detected in the dataset")
        else:
            st.info("Anomaly detection requires numeric columns in dataset")

    # ============================================================
    # TAB 3 — DATA EXPLORER
    # ============================================================
    with tab3:

        st.subheader("📄 Data Explorer")
        st.info("💡 Rows color-coded by risk. Confidence % shows ML certainty.")

        def highlight_risk(row):
            if row['Predicted Risk'] == 'HIGH':
                return ['background-color:#ff4d4d;color:white'] * len(row)
            elif row['Predicted Risk'] == 'MEDIUM':
                return ['background-color:#f1c40f;color:black'] * len(row)
            else:
                return ['background-color:#2ecc71;color:black'] * len(row)

        st.dataframe(
            filtered_df.style.apply(highlight_risk, axis=1),
            use_container_width=True
        )

        c1,c2,c3 = st.columns(3)
        with c1:
            st.download_button(
                "📥 Full Data", df_original.to_csv(index=False), "full.csv"
            )
        with c2:
            st.download_button(
                "📥 Filtered Data", filtered_df.to_csv(index=False), "filtered.csv"
            )
        with c3:
            if st.button("🔥 Sync Filtered to Firebase"):
                with st.spinner("Syncing..."):
                    try:
                        count = upload_dataframe_to_firestore(
                            filtered_df, collection="filtered_exports"
                        )
                        st.success(f"✅ {count} records synced to Firebase!")
                    except Exception as e:
                        st.error(f"Sync error: {e}")

    # ============================================================
    # TAB 4 — AI INSIGHTS
    # ============================================================
    with tab4:

        st.subheader("🧠 AI Insights (Groq GPT-OSS 120B)")
        show_model_performance()

        if not filtered_df.empty:

            high_c  = (filtered_df['Predicted Risk']=="HIGH").sum()
            med_c   = (filtered_df['Predicted Risk']=="MEDIUM").sum()
            low_c   = (filtered_df['Predicted Risk']=="LOW").sum()
            total_c = len(filtered_df)

            c1,c2,c3 = st.columns(3)
            c1.metric("🔴 HIGH",   high_c)
            c2.metric("🟡 MEDIUM", med_c)
            c3.metric("🟢 LOW",    low_c)

            if 'ai_summary' not in st.session_state:
                st.session_state.ai_summary = ""

            col1,col2 = st.columns(2)

            with col1:
                if st.button("📋 Summarize All Risks with AI"):
                    prompt = f"""
                    SAP Transport Dataset:
                    Total={total_c}, HIGH={high_c}, MEDIUM={med_c}, LOW={low_c}
                    Module breakdown:
                    {filtered_df.groupby(['module','Predicted Risk']).size().to_string()}
                    1) Overall summary 2) Riskiest modules 3) Top 3 recommendations
                    """
                    with st.spinner("AI analyzing..."):
                        result = ask_ai(prompt)
                    st.session_state.ai_summary = result
                    st.write(result)

            with col2:
                if st.button("📧 Generate Executive Email"):
                    top_mod = filtered_df.groupby('module')['Predicted Risk'].apply(
                        lambda x: (x=="HIGH").sum()
                    ).idxmax() if high_c > 0 else "N/A"

                    prompt = f"""
                    Write a professional executive email to the CTO:
                    - Total SAP Transports: {total_c}
                    - HIGH Risk: {high_c} ({round(high_c/total_c*100,1) if total_c else 0}%)
                    - MEDIUM Risk: {med_c}, LOW Risk: {low_c}
                    - Most at-risk module: {top_mod}
                    Format: Subject, greeting, bullet points, action items, sign-off.
                    """
                    with st.spinner("Drafting executive email..."):
                        email = ask_ai(prompt)
                    st.text_area("📧 Copy & send:", email, height=350)

            st.divider()

            # 7-Day Forecast
            st.markdown("### 🔮 7-Day Risk Trend Forecast")
            if st.button("📈 Predict Next 7 Days"):
                prod_count = len(
                    filtered_df[filtered_df['transport_stage']=="Production"]
                )
                top_module = filtered_df.groupby('module').size().idxmax()
                anomaly_c  = (filtered_df.get('Anomaly','Normal')=="⚠ Anomaly").sum()

                prompt = f"""
                Based on SAP transport data:
                HIGH={high_c}/{total_c}, Module={top_module},
                Production={prod_count}, Anomalies={anomaly_c}

                Predict 7-day risk trend as:
                - Day-by-day table (Day 1-7, Risk Level, Key Driver)
                - Overall 7-day outlook
                - Top action for this week
                """
                with st.spinner("Forecasting..."):
                    st.write(ask_ai(prompt))

            st.divider()

            # PDF Report
            st.markdown("### 📄 Export PDF Report")
            if st.button("🔄 Generate AI Summary for PDF"):
                with st.spinner("Generating..."):
                    prompt = f"""
                    SAP Risk Executive Report:
                    Total={total_c}, HIGH={high_c}, MEDIUM={med_c}, LOW={low_c}
                    Modules: {filtered_df.groupby(['module','Predicted Risk']).size().to_string()}
                    Write: executive summary, key risks, recommendations.
                    """
                    st.session_state.ai_summary = ask_ai(prompt)
                st.success("✅ Ready — click Download PDF")

            if st.session_state.ai_summary:
                pdf_buf = generate_pdf(
                    filtered_df,
                    st.session_state.ai_summary,
                    date_range_label
                )
                st.download_button(
                    "📥 Download PDF Report",
                    data=pdf_buf,
                    file_name="sap_risk_report.pdf",
                    mime="application/pdf"
                )

            st.divider()

            # Single Transport
            st.markdown("### 🔍 Analyze Specific Transport")
            t_ids      = filtered_df['transport_id'].astype(str).tolist()
            sel_id     = st.selectbox("Select Transport ID", t_ids)
            row        = filtered_df[
                filtered_df['transport_id'].astype(str)==sel_id
            ].iloc[0]

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("ID",     str(row['transport_id']))
            c2.metric("Module", row['module'])
            c3.metric("Stage",  row['transport_stage'])
            c4.metric("Risk",   row['Predicted Risk'])

            if row.get('Confidence %') not in [None,"N/A"]:
                st.metric("🎯 Confidence", f"{row['Confidence %']}%")

            if st.button("🔍 Generate AI Insight"):
                prompt = f"""
                SAP Transport {row['transport_id']}:
                Module={row['module']}, Risk={row['Predicted Risk']},
                Stage={row['transport_stage']},
                Confidence={row.get('Confidence %','N/A')}%
                Explain cause, business impact, mitigation steps.
                """
                with st.spinner("Analyzing..."):
                    insight = ask_ai(prompt)
                st.write(insight)

                # Save insight to Firebase
                try:
                    save_ai_insight(row['transport_id'], insight)
                    st.caption("✅ Insight saved to Firebase")
                except Exception as e:
                    st.caption(f"⚠ Could not save to Firebase: {e}")

            # Show saved insights from Firebase
            with st.expander("📚 View Saved AI Insights from Firebase"):
                try:
                    insights = fetch_ai_insights()
                    if insights:
                        for tid_key, data in insights.items():
                            st.markdown(f"**Transport:** {tid_key}")
                            st.write(data.get('insight',''))
                            st.markdown(
                                f"*Saved: {data.get('timestamp','')}*"
                            )
                            st.divider()
                    else:
                        st.info("No saved insights yet")
                except Exception as e:
                    st.error(f"Firebase fetch error: {e}")

        st.divider()

        # Chat with History
        st.subheader("💬 Ask AI — Chat with Memory")
        st.caption(
            f"Session: {st.session_state.session_id} — "
            "Chat history saved to Firebase"
        )

        if 'chat_history' not in st.session_state:
            # Try to load from Firebase
            try:
                saved = fetch_chat_history(st.session_state.session_id)
                st.session_state.chat_history = saved if saved else [{
                    "role":"system",
                    "content":"You are an expert SAP Basis consultant and transport risk analyst. Answer clearly and professionally."
                }]
            except:
                st.session_state.chat_history = [{
                    "role":"system",
                    "content":"You are an expert SAP Basis consultant and transport risk analyst."
                }]

        for msg in st.session_state.chat_history[1:]:
            if msg['role']=='user':
                st.chat_message("user").write(msg['content'])
            else:
                st.chat_message("assistant").write(msg['content'])

        query = st.chat_input("Ask anything about SAP transport risks…")
        if query:
            st.session_state.chat_history.append(
                {"role":"user","content":query}
            )
            st.chat_message("user").write(query)

            with st.spinner("AI thinking..."):
                reply = ask_ai_with_history(st.session_state.chat_history)

            st.session_state.chat_history.append(
                {"role":"assistant","content":reply}
            )
            st.chat_message("assistant").write(reply)

            # Save chat to Firebase
            try:
                save_chat_history(
                    st.session_state.session_id,
                    st.session_state.chat_history
                )
            except:
                pass

        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = [{
                "role":"system",
                "content":"You are an expert SAP Basis consultant."
            }]
            st.rerun()

    # ============================================================
    # TAB 5 — MANUAL PREDICTION
    # ============================================================
    with tab5:

        st.subheader("⚡ Real-Time Risk Prediction")

        c1,c2 = st.columns(2)
        with c1:
            module    = st.selectbox("Module",           ["FI","MM","SD","HR"])
            objects   = st.number_input("Objects Changed", 1, 10, 3)
            lines     = st.number_input("Lines Changed",   1, 500, 100)
            conflicts = st.number_input("Conflicts",       0, 5, 1)
        with c2:
            failures  = st.number_input("Failures",        0, 5, 1)
            stage     = st.selectbox("Stage",             ["Development","Quality","Production"])
            status    = st.selectbox("Status",            ["Approved","Pending","Rejected"])

        if st.button("🔮 Predict Risk"):

            new_df = pd.DataFrame([{
                "module":module, "objects_changed":objects,
                "lines_changed":lines, "conflicts":conflicts,
                "history_failures":failures, "transport_stage":stage,
                "change_request_status":status
            }])

            processed  = preprocess(new_df.copy())
            pred       = predict(model, processed)[0]
            conf, cls, proba_vals = get_confidence(processed)
            confidence = round(float(conf[0]),2) if conf is not None else None

            rc = cmap.get(pred,"#aaa")
            st.markdown(f"""
            <div style='background:{rc};
                 color:{"white" if pred=="HIGH" else "black"};
                 padding:20px;border-radius:15px;text-align:center;
                 font-size:1.8em;font-weight:bold;margin:10px 0'>
                {pred} RISK
                {f"— {confidence}% Confident" if confidence else ""}
            </div>""", unsafe_allow_html=True)

            if confidence and cls is not None:
                prob_df = pd.DataFrame({
                    'Risk Level':    cls,
                    'Probability %': (proba_vals[0]*100).round(2)
                })
                fig_pb = px.bar(
                    prob_df, x='Risk Level', y='Probability %',
                    title="🎯 Prediction Probability Breakdown",
                    color='Risk Level', color_discrete_map=cmap
                )
                st.plotly_chart(fig_pb, use_container_width=True)

            pct   = 90 if pred=="HIGH" else 60 if pred=="MEDIUM" else 30
            color = "#ff4d4d" if pred=="HIGH" else "#f1c40f" if pred=="MEDIUM" else "#2ecc71"
            st.markdown(risk_gauge(pct, color), unsafe_allow_html=True)

            fig_inp = px.bar(
                x=["Objects","Lines","Conflicts","Failures"],
                y=[objects, lines, conflicts, failures],
                title="Input Feature Values",
                color_discrete_sequence=[rc]
            )
            st.plotly_chart(fig_inp, use_container_width=True)

            pfig = plt.figure()
            plt.bar(["Objects","Lines","Conflicts","Failures"],
                    [objects,lines,conflicts,failures], color=rc)
            st.download_button(
                "📥 Export Chart", save_chart(pfig), "prediction.png"
            )

            if stage=="Production" and pred=="HIGH":
                st.error("🚨 DO NOT DEPLOY — HIGH-RISK PRODUCTION TRANSPORT!")

            # Save to Firebase
            pred_data = {
                "transport_id": f"MANUAL_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                "module":        module,
                "objects_changed":objects,
                "lines_changed":  lines,
                "conflicts":      conflicts,
                "history_failures":failures,
                "transport_stage":stage,
                "change_request_status":status,
                "predicted_risk": pred,
                "confidence":     float(confidence) if confidence else None,
                "timestamp":      datetime.datetime.now().isoformat()
            }
            try:
                save_prediction(pred_data)
                st.success("✅ Prediction saved to Firebase!")
            except Exception as e:
                st.warning(f"⚠ Could not save to Firebase: {e}")

            with st.expander("🤖 Get AI Explanation"):
                if st.button("Generate AI Insight"):
                    prompt = f"""
                    SAP transport predicted as {pred} risk ({confidence}% confidence).
                    Module={module}, Objects={objects}, Lines={lines},
                    Conflicts={conflicts}, Failures={failures},
                    Stage={stage}, Status={status}.
                    Explain why, business impact, 3 recommendations.
                    """
                    with st.spinner("AI analyzing..."):
                        st.write(ask_ai(prompt))

    # ============================================================
    # ============================================================
    # TAB 6 — WHAT-IF ANALYSIS
    # ============================================================
    with tab6:

        st.subheader("🔄 What-If Analysis")
        st.caption(
            "Compare a current transport scenario with an improved scenario "
            "using the SAME trained ML model."
        )

        st.markdown("### 📌 Scenario A — Current State")

        a1, a2 = st.columns(2)

        with a1:
            a_mod = st.selectbox(
                "Module (A)", ["FI", "MM", "SD", "HR"], key="am"
            )
            a_obj = st.number_input(
                "Objects Changed (A)", 1, 10, 5, key="ao"
            )
            a_lin = st.number_input(
                "Lines Changed (A)", 1, 500, 200, key="al"
            )
            a_con = st.number_input(
                "Conflicts (A)", 0, 5, 3, key="ac"
            )

        with a2:
            a_fail = st.number_input(
                "Historical Failures (A)", 0, 5, 2, key="af"
            )
            a_stg = st.selectbox(
                "Stage (A)",
                ["Development", "Quality", "Production"],
                key="as"
            )
            a_stat = st.selectbox(
                "Status (A)",
                ["Approved", "Pending", "Rejected"],
                key="ast"
            )

        st.markdown("### 🎯 Scenario B — Improved State")

        b1, b2 = st.columns(2)

        with b1:
            b_mod = st.selectbox(
                "Module (B)", ["FI", "MM", "SD", "HR"], key="bm"
            )
            b_obj = st.number_input(
                "Objects Changed (B)", 1, 10, 2, key="bo"
            )
            b_lin = st.number_input(
                "Lines Changed (B)", 1, 500, 50, key="bl"
            )
            b_con = st.number_input(
                "Conflicts (B)", 0, 5, 0, key="bc"
            )

        with b2:
            b_fail = st.number_input(
                "Historical Failures (B)", 0, 5, 0, key="bf"
            )
            b_stg = st.selectbox(
                "Stage (B)",
                ["Development", "Quality", "Production"],
                key="bs"
            )
            b_stat = st.selectbox(
                "Status (B)",
                ["Approved", "Pending", "Rejected"],
                key="bst"
            )

        if st.button("🔄 Compare Scenarios", key="compare_scenarios"):

            df_a = pd.DataFrame([{
                "module": a_mod,
                "objects_changed": a_obj,
                "lines_changed": a_lin,
                "conflicts": a_con,
                "history_failures": a_fail,
                "transport_stage": a_stg,
                "change_request_status": a_stat
            }])

            df_b = pd.DataFrame([{
                "module": b_mod,
                "objects_changed": b_obj,
                "lines_changed": b_lin,
                "conflicts": b_con,
                "history_failures": b_fail,
                "transport_stage": b_stg,
                "change_request_status": b_stat
            }])

            try:
                processed_a = preprocess(df_a.copy())
                processed_b = preprocess(df_b.copy())

                pred_a = predict(model, processed_a)[0]
                pred_b = predict(model, processed_b)[0]

                ca, classes_a, proba_a = get_confidence(processed_a)
                cb, classes_b, proba_b = get_confidence(processed_b)

                conf_a = (
                    round(float(ca[0]), 2)
                    if ca is not None else "N/A"
                )
                conf_b = (
                    round(float(cb[0]), 2)
                    if cb is not None else "N/A"
                )

                # Store the complete comparison so it survives Streamlit reruns.
                st.session_state.what_if_result = {
                    "pred_a": pred_a,
                    "pred_b": pred_b,
                    "conf_a": conf_a,
                    "conf_b": conf_b,
                    "df_a": df_a,
                    "df_b": df_b,
                    "proba_a": proba_a,
                    "proba_b": proba_b,
                    "classes_a": classes_a,
                    "classes_b": classes_b
                }

                # Clear old AI explanation when a new comparison is made.
                st.session_state.what_if_ai = ""

            except Exception as e:
                st.error(
                    f"❌ Could not compare the scenarios: "
                    f"{type(e).__name__}: {e}"
                )

        # ------------------------------------------------------------
        # DISPLAY LAST COMPARISON
        # ------------------------------------------------------------
        if "what_if_result" in st.session_state:

            result = st.session_state.what_if_result

            pred_a = result["pred_a"]
            pred_b = result["pred_b"]
            conf_a = result["conf_a"]
            conf_b = result["conf_b"]

            df_a = result["df_a"]
            df_b = result["df_b"]

            proba_a = result["proba_a"]
            proba_b = result["proba_b"]
            classes_a = result["classes_a"]
            classes_b = result["classes_b"]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📌 Scenario A — Current")

                rc_a = cmap.get(pred_a, "#aaa")

                st.markdown(
                    f"""
                    <div style='background:{rc_a};
                         color:{"white" if pred_a=="HIGH" else "black"};
                         padding:22px;border-radius:14px;
                         text-align:center;font-size:1.6em;
                         font-weight:bold'>
                        {pred_a} RISK<br>
                        <span style='font-size:0.62em'>
                            Model Confidence: {conf_a}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                pct_a = (
                    90 if pred_a == "HIGH"
                    else 60 if pred_a == "MEDIUM"
                    else 30
                )

                st.markdown(
                    risk_gauge(pct_a, rc_a),
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown("#### 🎯 Scenario B — Improved")

                rc_b = cmap.get(pred_b, "#aaa")

                st.markdown(
                    f"""
                    <div style='background:{rc_b};
                         color:{"white" if pred_b=="HIGH" else "black"};
                         padding:22px;border-radius:14px;
                         text-align:center;font-size:1.6em;
                         font-weight:bold'>
                        {pred_b} RISK<br>
                        <span style='font-size:0.62em'>
                            Model Confidence: {conf_b}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                pct_b = (
                    90 if pred_b == "HIGH"
                    else 60 if pred_b == "MEDIUM"
                    else 30
                )

                st.markdown(
                    risk_gauge(pct_b, rc_b),
                    unsafe_allow_html=True
                )

            # --------------------------------------------------------
            # RISK COMPARISON BAR CHART
            # --------------------------------------------------------
            st.divider()
            st.markdown("### 📊 Scenario Risk Comparison")

            risk_score = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}

            score_a = risk_score.get(pred_a, 0)
            score_b = risk_score.get(pred_b, 0)

            comparison_df = pd.DataFrame({
                "Scenario": ["Scenario A", "Scenario B"],
                "Risk Score": [score_a, score_b],
                "Risk Level": [pred_a, pred_b],
                "Confidence %": [
                    float(conf_a) if conf_a != "N/A" else 0,
                    float(conf_b) if conf_b != "N/A" else 0
                ]
            })

            fig_compare = px.bar(
                comparison_df,
                x="Scenario",
                y="Risk Score",
                text="Risk Level",
                title="Risk Score Comparison  •  HIGH=3, MEDIUM=2, LOW=1",
                range_y=[0, 3.4],
                color="Risk Level",
                color_discrete_map=cmap
            )

            fig_compare.update_traces(
                textposition="outside"
            )

            fig_compare.update_layout(
                yaxis_title="Risk Score",
                xaxis_title="Scenario",
                showlegend=True
            )

            st.plotly_chart(
                fig_compare,
                use_container_width=True
            )

            # --------------------------------------------------------
            # FEATURE CHANGE CHART
            # --------------------------------------------------------
            st.markdown("### 📈 What Changed from A → B?")

            feature_df = pd.DataFrame({
                "Feature": [
                    "Objects Changed",
                    "Lines Changed",
                    "Conflicts",
                    "Historical Failures"
                ],
                "Scenario A": [
                    a_obj, a_lin, a_con, a_fail
                ],
                "Scenario B": [
                    b_obj, b_lin, b_con, b_fail
                ]
            })

            feature_long = feature_df.melt(
                id_vars="Feature",
                var_name="Scenario",
                value_name="Value"
            )

            fig_features = px.bar(
                feature_long,
                x="Feature",
                y="Value",
                color="Scenario",
                barmode="group",
                text="Value",
                title="Input Feature Comparison"
            )

            fig_features.update_traces(
                textposition="outside"
            )

            st.plotly_chart(
                fig_features,
                use_container_width=True
            )

            # --------------------------------------------------------
            # DETAILED NUMERIC DELTA
            # --------------------------------------------------------
            delta_rows = []

            for label, a_val, b_val in [
                ("Objects Changed", a_obj, b_obj),
                ("Lines Changed", a_lin, b_lin),
                ("Conflicts", a_con, b_con),
                ("Historical Failures", a_fail, b_fail)
            ]:
                delta_rows.append({
                    "Feature": label,
                    "Scenario A": a_val,
                    "Scenario B": b_val,
                    "Change (B-A)": b_val - a_val
                })

            delta_df = pd.DataFrame(delta_rows)

            st.dataframe(
                delta_df,
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------------------------------
            # DECISION SUMMARY
            # --------------------------------------------------------
            delta_risk = score_a - score_b

            if delta_risk > 0:
                st.success(
                    f"✅ Risk REDUCED from **{pred_a} → {pred_b}**. "
                    "Scenario B is safer according to the trained model."
                )
            elif delta_risk < 0:
                st.error(
                    f"⚠ Risk INCREASED from **{pred_a} → {pred_b}**. "
                    "Scenario A is safer according to the trained model."
                )
            else:
                st.info(
                    f"➡ Risk level is unchanged: both scenarios predict "
                    f"**{pred_a}**."
                )

            # --------------------------------------------------------
            # WHY SCENARIO B IS SAFER
            # --------------------------------------------------------
            st.markdown("### 🧠 Why Is Scenario B Safer?")

            if delta_risk > 0:
                reasons = []

                if b_obj < a_obj:
                    reasons.append(
                        f"Objects changed decreased from **{a_obj} → {b_obj}**."
                    )

                if b_lin < a_lin:
                    reasons.append(
                        f"Lines changed decreased from **{a_lin} → {b_lin}**."
                    )

                if b_con < a_con:
                    reasons.append(
                        f"Conflicts decreased from **{a_con} → {b_con}**."
                    )

                if b_fail < a_fail:
                    reasons.append(
                        f"Historical failures decreased from "
                        f"**{a_fail} → {b_fail}**."
                    )

                if a_stg != b_stg:
                    reasons.append(
                        f"Transport stage changed from **{a_stg} → {b_stg}**."
                    )

                if a_stat != b_stat:
                    reasons.append(
                        f"Change-request status changed from "
                        f"**{a_stat} → {b_stat}**."
                    )

                if reasons:
                    for reason in reasons:
                        st.markdown(f"• {reason}")
                else:
                    st.markdown(
                        "• The input values shown are unchanged, but the "
                        "model produced a lower risk for Scenario B."
                    )

                st.caption(
                    "These are model-based indicators, not proof of causality. "
                    "The final risk decision should be validated in the actual "
                    "SAP landscape."
                )

            elif delta_risk == 0:
                st.info(
                    "Scenario B is not safer because the model assigned the "
                    "same risk level to both scenarios. Review the feature "
                    "changes and model confidence."
                )

            else:
                st.warning(
                    "Scenario B is not safer according to the model. "
                    "Review the changes before deployment."
                )

            # --------------------------------------------------------
            # AI EXPLANATION
            # --------------------------------------------------------
            st.markdown("### 🤖 AI Explanation — Scenario Comparison")

            if st.button(
                "🧠 Explain Why Scenario B Is Safer",
                key="what_if_ai_button"
            ):
                ai_prompt = f"""
You are an SAP Basis and transport-risk analyst.

Compare these two scenarios using the ML model result.

SCENARIO A — CURRENT
Module: {a_mod}
Objects Changed: {a_obj}
Lines Changed: {a_lin}
Conflicts: {a_con}
Historical Failures: {a_fail}
Stage: {a_stg}
Status: {a_stat}
Predicted Risk: {pred_a}
Model Confidence: {conf_a}%

SCENARIO B — IMPROVED
Module: {b_mod}
Objects Changed: {b_obj}
Lines Changed: {b_lin}
Conflicts: {b_con}
Historical Failures: {b_fail}
Stage: {b_stg}
Status: {b_stat}
Predicted Risk: {pred_b}
Model Confidence: {conf_b}%

Explain:
1. Whether Scenario B is safer according to the model.
2. Exactly which input changes distinguish B from A.
3. Which changes are the strongest practical risk-reduction signals.
4. Why the model's predicted risk changed or stayed the same.
5. What should be validated before a Production deployment.

Do not claim that one feature caused the prediction unless the model evidence
supports that conclusion. Keep the explanation professional and concise.
"""

                with st.spinner("🤖 AI explaining the scenario difference..."):
                    st.session_state.what_if_ai = ask_ai(ai_prompt)

            if st.session_state.get("what_if_ai"):
                st.info(st.session_state.what_if_ai)

            # --------------------------------------------------------
            # PROBABILITY BREAKDOWN
            # --------------------------------------------------------
            if (
                proba_a is not None
                and proba_b is not None
                and classes_a is not None
                and classes_b is not None
            ):
                st.markdown("### 🎯 Model Probability Breakdown")

                prob_a = pd.DataFrame({
                    "Risk Level": classes_a,
                    "Scenario A": (proba_a[0] * 100).round(2)
                })

                prob_b = pd.DataFrame({
                    "Risk Level": classes_b,
                    "Scenario B": (proba_b[0] * 100).round(2)
                })

                prob_df = pd.merge(
                    prob_a,
                    prob_b,
                    on="Risk Level",
                    how="outer"
                ).fillna(0)

                prob_long = prob_df.melt(
                    id_vars="Risk Level",
                    var_name="Scenario",
                    value_name="Probability %"
                )

                fig_prob = px.bar(
                    prob_long,
                    x="Risk Level",
                    y="Probability %",
                    color="Scenario",
                    barmode="group",
                    text="Probability %",
                    title="Model Confidence Distribution"
                )

                fig_prob.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside"
                )

                st.plotly_chart(
                    fig_prob,
                    use_container_width=True
                )
    # TAB 7 — PREDICTION HISTORY (FIREBASE)
    # ============================================================
    with tab7:

        st.subheader("📜 Prediction History")
        st.caption("All manual predictions saved to Firebase Firestore")

        if st.button("🔄 Refresh from Firebase"):
            st.rerun()

        try:
            history_df = fetch_predictions()

            if not history_df.empty:
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Total Predictions", len(history_df))
                c2.metric("HIGH Risk",
                    (history_df['predicted_risk']=="HIGH").sum()
                    if 'predicted_risk' in history_df.columns else 0)
                c3.metric("MEDIUM Risk",
                    (history_df['predicted_risk']=="MEDIUM").sum()
                    if 'predicted_risk' in history_df.columns else 0)
                c4.metric("LOW Risk",
                    (history_df['predicted_risk']=="LOW").sum()
                    if 'predicted_risk' in history_df.columns else 0)

                # Color coded
                if 'predicted_risk' in history_df.columns:
                    def highlight_pred(row):
                        if row.get('predicted_risk') == 'HIGH':
                            return ['background-color:#ff4d4d;color:white']*len(row)
                        elif row.get('predicted_risk') == 'MEDIUM':
                            return ['background-color:#f1c40f;color:black']*len(row)
                        else:
                            return ['background-color:#2ecc71;color:black']*len(row)

                    st.dataframe(
                        history_df.style.apply(highlight_pred, axis=1),
                        use_container_width=True
                    )

                    # Chart
                    fig_h = px.bar(
                        history_df['predicted_risk'].value_counts().reset_index(),
                        x='predicted_risk', y='count',
                        title="Prediction History Distribution",
                        color='predicted_risk',
                        color_discrete_map=cmap
                    )
                    st.plotly_chart(fig_h, use_container_width=True)
                else:
                    st.dataframe(history_df, use_container_width=True)

                st.download_button(
                    "📥 Export Prediction History",
                    history_df.to_csv(index=False),
                    "prediction_history.csv"
                )

                # AI Summary of history
                if st.button("🤖 AI Analyze Prediction History"):
                    if 'predicted_risk' in history_df.columns:
                        prompt = f"""
                        Analyze this SAP transport prediction history:
                        Total predictions: {len(history_df)}
                        HIGH: {(history_df['predicted_risk']=="HIGH").sum()}
                        MEDIUM: {(history_df['predicted_risk']=="MEDIUM").sum()}
                        LOW: {(history_df['predicted_risk']=="LOW").sum()}

                        Identify patterns, trends, and risk insights
                        from these historical predictions.
                        """
                        with st.spinner("AI analyzing history..."):
                            st.write(ask_ai(prompt))

            else:
                st.info(
                    "No prediction history yet. "
                    "Use Manual Prediction tab to generate predictions."
                )

        except Exception as e:
            st.error(f"Firebase fetch error: {e}")
            st.info("Make sure Firebase is connected and firebase_key.json is correct")


    # ============================================================
    # TAB 8 — BUSINESS IMPACT & ROI
    # ============================================================
    with tab8:

        st.markdown("""
        <div style='background:linear-gradient(135deg,#0f0c29,#302b63);
             padding:25px;border-radius:15px;margin-bottom:20px;
             border:1px solid #764ba2'>
            <h2 style='color:white;margin:0'>💰 Business Impact & ROI Calculator</h2>
            <p style='color:#b39ddb;margin-top:8px'>
                Quantify the financial value of AI-powered SAP transport risk prevention
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ---- ROI INPUTS ----
        st.markdown("### ⚙️ Configure Your Business Parameters")
        st.caption("Adjust these values to match your company's actual costs")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💸 Cost Parameters")
            cost_per_incident = st.number_input(
                "Average cost per SAP incident (₹)",
                min_value=10000,
                max_value=10000000,
                value=50000,
                step=10000,
                help="Include downtime, IT overtime, business loss"
            )
            downtime_hours = st.number_input(
                "Average downtime per incident (hours)",
                min_value=1,
                max_value=72,
                value=4,
                help="Average hours to resolve a failed transport"
            )
            hourly_business_loss = st.number_input(
                "Business loss per hour (₹)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=5000,
                help="Revenue/productivity lost per hour of downtime"
            )

        with col2:
            st.markdown("#### 📊 Prevention Parameters")
            prevention_rate = st.slider(
                "AI Prevention Rate (%)",
                min_value=50,
                max_value=95,
                value=60,
                help="% of HIGH risk transports caught before failure"
            )
            transports_per_year = st.number_input(
                "Transports deployed per year",
                min_value=100,
                max_value=10000000,
                value=1000,
                step=100,
                help="Estimated annual transport volume"
            )
            implementation_cost = st.number_input(
                "Platform implementation cost (₹)",
                min_value=0,
                max_value=5000000,
                value=200000,
                step=50000,
                help="One-time cost to deploy this platform"
            )

        st.divider()

        # ---- ROI CALCULATIONS ----
        high_risk_rate     = high_n / total_n if total_n > 0 else 0
        annual_high_risk   = int(transports_per_year * high_risk_rate)
        incidents_prevented= int(annual_high_risk * prevention_rate / 100)
        downtime_prevented = incidents_prevented * downtime_hours
        direct_savings     = incidents_prevented * cost_per_incident
        downtime_savings   = downtime_prevented * hourly_business_loss
        total_savings      = direct_savings + downtime_savings
        net_roi            = total_savings - implementation_cost
        roi_percentage     = ((net_roi / implementation_cost) * 100) if implementation_cost > 0 else 0
        payback_days       = int((implementation_cost / (total_savings / 365))) if total_savings > 0 else 0

        # ---- ROI DISPLAY ----
        st.markdown("### 📊 ROI Analysis Results")

        r1,r2,r3,r4 = st.columns(4)
        r1.metric(
            "💰 Total Annual Savings",
            f"₹{total_savings:,.0f}",
            delta="Per Year"
        )
        r2.metric(
            "📈 ROI Percentage",
            f"{roi_percentage:,.0f}%",
            delta="Return on Investment"
        )
        r3.metric(
            "⚡ Payback Period",
            f"{payback_days} days",
            delta="To recover investment"
        )
        r4.metric(
            "🛡 Incidents Prevented",
            f"{incidents_prevented:,}",
            delta=f"{prevention_rate}% prevention rate"
        )

        st.divider()

        # ---- SAVINGS BREAKDOWN ----
        st.markdown("### 💵 Savings Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            # Savings breakdown chart
            savings_data = {
                'Category': [
                    'Direct Incident Savings',
                    'Downtime Prevention',
                    'Implementation Cost'
                ],
                'Amount (₹)': [
                    direct_savings,
                    downtime_savings,
                    -implementation_cost
                ]
            }
            fig_roi = px.bar(
                savings_data,
                x='Category',
                y='Amount (₹)',
                title="Annual Savings vs Cost Breakdown",
                color='Category',
                color_discrete_sequence=['#2ecc71','#3498db','#e74c3c']
            )
            st.plotly_chart(fig_roi, use_container_width=True)

        with col2:
            # Risk distribution impact
            impact_data = {
                'Risk Type': ['HIGH Risk', 'MEDIUM Risk', 'LOW Risk'],
                'Count':     [high_n, med_n, low_n]
            }
            fig_impact = px.pie(
                impact_data,
                values='Count',
                names='Risk Type',
                title="Current Risk Distribution",
                color='Risk Type',
                color_discrete_map={
                    'HIGH Risk':'#ff4d4d',
                    'MEDIUM Risk':'#f1c40f',
                    'LOW Risk':'#2ecc71'
                }
            )
            st.plotly_chart(fig_impact, use_container_width=True)

        st.divider()

        # ---- BUSINESS IMPACT SUMMARY ----
        st.markdown("### 🏢 Business Impact Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div style='background:#1a1a2e;border:1px solid #2ecc71;
                 border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:2em'>🛡️</div>
                <h3 style='color:#2ecc71;margin:8px 0'>
                    {incidents_prevented:,}
                </h3>
                <p style='color:#aaa;margin:0'>
                    Production incidents prevented annually
                </p>
            </div>""", unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style='background:#1a1a2e;border:1px solid #3498db;
                 border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:2em'>⏱️</div>
                <h3 style='color:#3498db;margin:8px 0'>
                    {downtime_prevented:,} hrs
                </h3>
                <p style='color:#aaa;margin:0'>
                    Downtime prevented annually
                </p>
            </div>""", unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style='background:#1a1a2e;border:1px solid #f1c40f;
                 border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:2em'>💰</div>
                <h3 style='color:#f1c40f;margin:8px 0'>
                    ₹{net_roi:,.0f}
                </h3>
                <p style='color:#aaa;margin:0'>
                    Net annual savings after costs
                </p>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ---- SLA COMPLIANCE ----
        st.markdown("### 📋 SLA Compliance Analysis")

        sla_breach    = filtered_df[
            (filtered_df['Predicted Risk']=="HIGH") &
            (filtered_df['transport_stage']=="Production")
        ]
        compliance_score = round(
            (1 - len(sla_breach)/total_n)*100, 1
        ) if total_n > 0 else 100

        s1,s2,s3,s4 = st.columns(4)
        s1.metric("📋 SLA Compliance Score",  f"{compliance_score}%")
        s2.metric("🚨 SLA Breach Risk",        len(sla_breach))
        s3.metric("✅ Compliant Transports",
                  total_n - len(sla_breach))
        s4.metric("🎯 Target Compliance",       "95%",
                  delta=f"{round(compliance_score-95,1)}%")

        # Compliance gauge color
        comp_color = "#2ecc71" if compliance_score >= 95                      else "#f1c40f" if compliance_score >= 80                      else "#ff4d4d"

        st.markdown(f"""
        <div style='background:#1a1a2e;border:1px solid {comp_color};
             border-radius:12px;padding:20px;margin:15px 0'>
            <div style='display:flex;justify-content:space-between;
                 align-items:center'>
                <div>
                    <h3 style='color:white;margin:0'>
                        SLA Compliance Status
                    </h3>
                    <p style='color:#aaa;margin:5px 0'>
                        Based on HIGH risk transports in Production
                    </p>
                </div>
                <div style='font-size:2.5em;font-weight:bold;
                     color:{comp_color}'>
                    {compliance_score}%
                </div>
            </div>
            <div style='background:#333;border-radius:10px;
                 height:20px;margin-top:15px'>
                <div style='background:{comp_color};
                     width:{min(compliance_score,100)}%;
                     height:20px;border-radius:10px;
                     transition:width 0.5s'>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ---- MODULE IMPACT ----
        st.markdown("### 📦 Module-wise Business Impact")

        module_impact = []
        for mod in filtered_df['module'].unique():
            mdf       = filtered_df[filtered_df['module']==mod]
            mod_high  = (mdf['Predicted Risk']=="HIGH").sum()
            mod_total = len(mdf)
            mod_risk  = round(mod_high/mod_total*100,1) if mod_total>0 else 0
            mod_cost  = int(mod_high * prevention_rate/100 * cost_per_incident)
            module_impact.append({
                'Module':          mod,
                'Total':           mod_total,
                'HIGH Risk':       mod_high,
                'Risk %':          mod_risk,
                'Potential Savings (₹)': mod_cost
            })

        impact_df = pd.DataFrame(module_impact).sort_values(
            'Potential Savings (₹)', ascending=False
        )
        st.dataframe(impact_df, use_container_width=True)

        fig_mod = px.bar(
            impact_df,
            x='Module',
            y='Potential Savings (₹)',
            title="Potential Savings by Module",
            color='Risk %',
            color_continuous_scale=['green','yellow','red']
        )
        st.plotly_chart(fig_mod, use_container_width=True)

        st.divider()

        # ---- AI BUSINESS SUMMARY ----
        st.markdown("### 🤖 AI-Generated Business Impact Report")

        if st.button("📊 Generate AI Business Impact Summary"):
            prompt = f"""
            Generate a professional business impact report for SAP Transport Risk AI Platform:

            Current Data:
            - Total Transports Analyzed: {total_n:,}
            - HIGH Risk Detected: {high_n:,} ({round(high_n/total_n*100,1) if total_n else 0}%)
            - Critical (Production+HIGH): {len(sla_breach):,}
            - SLA Compliance Score: {compliance_score}%

            Financial Impact:
            - Annual Incidents Prevented: {incidents_prevented:,}
            - Total Annual Savings: ₹{total_savings:,.0f}
            - ROI: {roi_percentage:,.0f}%
            - Payback Period: {payback_days} days
            - Downtime Prevented: {downtime_prevented:,} hours/year

            Module Risk:
            {impact_df[['Module','HIGH Risk','Risk %']].to_string()}

            Write a compelling 1-page executive business impact report including:
            1. Executive Summary
            2. Key Risk Findings
            3. Financial Benefits
            4. Strategic Recommendations
            5. Conclusion with ROI justification

            Make it professional and suitable for C-suite presentation.
            """
            with st.spinner("AI generating business report..."):
                report = ask_ai(prompt)
            st.write(report)

            # Save for PDF
            st.session_state.ai_summary = report

            # Generate PDF button
            pdf_buf = generate_pdf(
                filtered_df, report,
                f"ROI Analysis — {datetime.datetime.now().strftime('%Y-%m-%d')}"
            )
            st.download_button(
                "📥 Download Business Impact PDF",
                data=pdf_buf,
                file_name="sap_business_impact_report.pdf",
                mime="application/pdf"
            )
    # ============================================================
    # TAB 9 — WHOLE-DATASET DEPLOYMENT DECISION GATE
    # ============================================================
    with tab9:

        st.markdown("""
        <div style='background:linear-gradient(135deg,#0f0c29,#302b63);
             padding:25px;border-radius:15px;margin-bottom:20px;
             border:1px solid #764ba2'>
            <h2 style='color:white;margin:0'>
                🚦 AI Deployment Decision Gate
            </h2>
            <p style='color:#b39ddb;margin-top:8px'>
                Whole-dataset risk assessment for deployment readiness
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.info(
            "🔒 This gate evaluates the complete loaded dataset "
            "and is independent of the sidebar filters."
        )

        # --------------------------------------------------------
        # WHOLE DATASET
        # --------------------------------------------------------

        gate_df = df_original.copy()

        gate_total = len(gate_df)

        gate_high = (
            gate_df["Predicted Risk"] == "HIGH"
        ).sum()

        gate_medium = (
            gate_df["Predicted Risk"] == "MEDIUM"
        ).sum()

        gate_low = (
            gate_df["Predicted Risk"] == "LOW"
        ).sum()

        gate_high_pct = (
            gate_high / gate_total * 100
            if gate_total > 0 else 0
        )

        # --------------------------------------------------------
        # PRODUCTION HIGH-RISK TRANSPORTS
        # --------------------------------------------------------

        gate_production_high = gate_df[
            (gate_df["transport_stage"] == "Production") &
            (gate_df["Predicted Risk"] == "HIGH")
        ]

        production_high_count = len(gate_production_high)

        production_count = (
            gate_df["transport_stage"] == "Production"
        ).sum()

        production_high_pct = (
            production_high_count / production_count * 100
            if production_count > 0 else 0
        )

        # --------------------------------------------------------
        # CONFLICT / FAILURE SIGNALS
        # --------------------------------------------------------

        conflict_count = (
            gate_df["conflicts"] > 0
        ).sum()

        failure_count = (
            gate_df["history_failures"] > 0
        ).sum()

        anomaly_count = (
            gate_df["Anomaly"] == "⚠ Anomaly"
        ).sum() if "Anomaly" in gate_df.columns else 0

        # --------------------------------------------------------
        # RISK CONCENTRATION
        # --------------------------------------------------------

        if gate_total > 0:

            high_risk_rate = gate_high / gate_total

            critical_rate = (
                production_high_count / gate_total
            )

        else:

            high_risk_rate = 0
            critical_rate = 0

        # --------------------------------------------------------
        # DEPLOYMENT POLICY
        #
        # Demo policy:
        #
        # NO-GO:
        #   Production HIGH risk exists
        #   OR whole dataset HIGH risk >= 20%
        #
        # CONDITIONAL GO:
        #   HIGH risk exists but critical production
        #   exposure is limited
        #
        # GO:
        #   No HIGH risk and no critical production exposure
        # --------------------------------------------------------

        if production_high_count > 0 or high_risk_rate >= 0.20:

            gate_decision = "NO-GO"
            gate_color = "#e74c3c"
            gate_icon = "🛑"

            gate_reason = (
                "High-risk deployment exposure detected. "
                "Production deployment should be blocked "
                "until remediation and validation are completed."
            )

        elif gate_high > 0:

            gate_decision = "CONDITIONAL GO"
            gate_color = "#f1c40f"
            gate_icon = "⚠️"

            gate_reason = (
                "High-risk transports exist in the dataset, "
                "but no HIGH-risk Production transport was detected. "
                "Additional validation is recommended before deployment."
            )

        else:

            gate_decision = "GO"
            gate_color = "#2ecc71"
            gate_icon = "✅"

            gate_reason = (
                "No HIGH-risk transports were detected. "
                "Dataset is currently within the configured "
                "deployment risk policy."
            )

# --------------------------------------------------------
# MAIN DECISION CARD
# --------------------------------------------------------

        st.markdown("### 🎯 Deployment Decision")

        decision_card_html = f"""
        <div style="
            background:#1a1a2e;
            border:3px solid {gate_color};
            border-radius:18px;
            padding:30px;
            text-align:center;
            margin:15px 0 25px 0;
        ">

            <div style="font-size:3em; margin-bottom:10px;">
                {gate_icon}
            </div>

            <div style="
                font-size:2.5em;
                font-weight:bold;
                color:{gate_color};
                margin:10px 0;
            ">
                {gate_decision}
            </div>

            <div style="
                color:#ddd;
                font-size:1.05em;
                max-width:850px;
                margin:auto;
                line-height:1.6;
            ">
                {gate_reason}
            </div>

        </div>
        """

        st.html(decision_card_html)
        # --------------------------------------------------------
        # EXECUTIVE KPIs
        # --------------------------------------------------------

        g1,g2,g3,g4,g5 = st.columns(5)

        g1.metric(
            "📦 Total Transports",
            f"{gate_total:,}"
        )

        g2.metric(
            "🔴 HIGH Risk",
            f"{gate_high:,}",
            delta=f"{gate_high_pct:.1f}%"
        )

        g3.metric(
            "🚨 Production HIGH",
            f"{production_high_count:,}"
        )

        g4.metric(
            "⚠️ Anomalies",
            f"{anomaly_count:,}"
        )

        g5.metric(
            "💥 Conflict Exposure",
            f"{conflict_count:,}"
        )

        st.divider()

        # --------------------------------------------------------
        # RISK DISTRIBUTION
        # --------------------------------------------------------

        st.markdown("### 📊 Whole-Dataset Risk Profile")

        risk_gate_df = pd.DataFrame({
            "Risk Level": [
                "HIGH",
                "MEDIUM",
                "LOW"
            ],
            "Transport Count": [
                gate_high,
                gate_medium,
                gate_low
            ]
        })

        col1, col2 = st.columns(2)

        with col1:

            fig_gate = px.bar(
                risk_gate_df,
                x="Risk Level",
                y="Transport Count",
                color="Risk Level",
                title="Complete Dataset Risk Distribution",
                color_discrete_map={
                    "HIGH": "#ff4d4d",
                    "MEDIUM": "#f1c40f",
                    "LOW": "#2ecc71"
                }
            )

            st.plotly_chart(
                fig_gate,
                use_container_width=True
            )

        with col2:

            fig_gate_pie = px.pie(
                risk_gate_df,
                values="Transport Count",
                names="Risk Level",
                title="Risk Composition — Entire Dataset",
                color="Risk Level",
                color_discrete_map={
                    "HIGH": "#ff4d4d",
                    "MEDIUM": "#f1c40f",
                    "LOW": "#2ecc71"
                }
            )

            st.plotly_chart(
                fig_gate_pie,
                use_container_width=True
            )

        # --------------------------------------------------------
        # DECISION RULES
        # --------------------------------------------------------

        st.divider()

        st.markdown("### 🧠 Decision Gate Rules")

        rule1 = (
            "🔴 BLOCK"
            if production_high_count > 0
            else "🟢 PASS"
        )

        rule2 = (
            "🔴 BLOCK"
            if high_risk_rate >= 0.20
            else "🟢 PASS"
        )

        rule3 = (
            "⚠️ REVIEW"
            if anomaly_count > 0
            else "🟢 PASS"
        )

        rules_df = pd.DataFrame({
            "Control": [
                "Production HIGH-Risk Exposure",
                "Overall HIGH-Risk Concentration",
                "AI Anomaly Exposure"
            ],
            "Observed": [
                f"{production_high_count} transports",
                f"{gate_high} ({gate_high_pct:.1f}%)",
                f"{anomaly_count} anomalies"
            ],
            "Gate Status": [
                rule1,
                rule2,
                rule3
            ]
        })

        st.dataframe(
            rules_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------------
        # TOP RISK MODULES
        # --------------------------------------------------------

        st.divider()

        st.markdown("### 📦 Risk Concentration by Module")

        module_gate = (
            gate_df
            .groupby("module")
            .agg(
                Total=("transport_id", "count"),
                HIGH=("Predicted Risk",
                      lambda x: (x == "HIGH").sum()),
                MEDIUM=("Predicted Risk",
                        lambda x: (x == "MEDIUM").sum()),
                LOW=("Predicted Risk",
                     lambda x: (x == "LOW").sum())
            )
            .reset_index()
        )

        module_gate["HIGH Risk %"] = (
            module_gate["HIGH"] /
            module_gate["Total"] * 100
        ).round(1)

        module_gate = module_gate.sort_values(
            "HIGH Risk %",
            ascending=False
        )

        st.dataframe(
            module_gate,
            use_container_width=True,
            hide_index=True
        )

        fig_module_gate = px.bar(
            module_gate,
            x="module",
            y="HIGH Risk %",
            title="HIGH-Risk Concentration by SAP Module",
            text="HIGH Risk %"
        )

        fig_module_gate.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig_module_gate,
            use_container_width=True
        )

        # --------------------------------------------------------
        # TOP NO-GO DRIVERS
        # --------------------------------------------------------

        st.divider()

        st.markdown("### 🔥 Why Is the Gate Blocking Deployment?")

        drivers = []

        if production_high_count > 0:
            drivers.append({
                "Priority": "CRITICAL",
                "Driver": "HIGH-risk Production transports",
                "Impact": f"{production_high_count} transports"
            })

        if gate_high > 0:
            drivers.append({
                "Priority": "HIGH",
                "Driver": "Overall HIGH-risk concentration",
                "Impact": f"{gate_high_pct:.1f}% of dataset"
            })

        if conflict_count > 0:
            drivers.append({
                "Priority": "HIGH",
                "Driver": "Transport conflicts detected",
                "Impact": f"{conflict_count} transports"
            })

        if failure_count > 0:
            drivers.append({
                "Priority": "HIGH",
                "Driver": "Historical transport failures",
                "Impact": f"{failure_count} transports"
            })

        if anomaly_count > 0:
            drivers.append({
                "Priority": "MEDIUM",
                "Driver": "AI-detected anomalous patterns",
                "Impact": f"{anomaly_count} transports"
            })

        if not drivers:

            st.success(
                "✅ No major blocking risk drivers detected."
            )

        else:

            drivers_df = pd.DataFrame(drivers)

            st.dataframe(
                drivers_df,
                use_container_width=True,
                hide_index=True
            )

        # --------------------------------------------------------
        # ACTION PLAN
        # --------------------------------------------------------

        st.divider()

        st.markdown("### 🛠️ Recommended Action")

        if gate_decision == "NO-GO":

            st.error("""
            ### 🛑 Deployment should be blocked

            **Recommended sequence:**

            1. Identify HIGH-risk Production transports.
            2. Resolve conflicts and historical failure conditions.
            3. Validate risky transports in Quality.
            4. Re-run the ML risk assessment.
            5. Re-check the Deployment Decision Gate.
            6. Proceed only after the gate reaches an acceptable state.
            """)

        elif gate_decision == "CONDITIONAL GO":

            st.warning("""
            ### ⚠️ Conditional deployment

            **Recommended sequence:**

            1. Review all HIGH-risk transports.
            2. Validate them in Quality.
            3. Review detected anomalies.
            4. Re-run the risk assessment.
            5. Obtain deployment approval.
            """)

        else:

            st.success("""
            ### ✅ Deployment can proceed toward approval

            No HIGH-risk transports were detected by the current
            model and configured demo policy.

            Continue with standard validation and organizational
            approval procedures.
            """)

        # --------------------------------------------------------
        # --------------------------------------------------------
        # AI EXECUTIVE DECISION
        # --------------------------------------------------------

        st.divider()
        st.markdown("### 🤖 AI Executive Decision Explanation")

        if "gate_ai_explanation" not in st.session_state:
            st.session_state.gate_ai_explanation = ""

        if st.button(
            "🧠 Explain Deployment Decision with AI",
            key="ai_deployment_gate"
        ):

            gate_prompt = f"""
You are an SAP transport risk decision analyst.

Analyze the WHOLE DATASET assessment below.

Total transports: {gate_total}
HIGH risk: {gate_high} ({gate_high_pct:.1f}%)
MEDIUM risk: {gate_medium}
LOW risk: {gate_low}

Production transports: {production_count}
HIGH-risk Production transports: {production_high_count}

Transports with conflicts: {conflict_count}
Transports with historical failures: {failure_count}
AI anomalies: {anomaly_count}

Current deployment gate decision:
{gate_decision}

Explain clearly:

1. Why the Deployment Decision Gate produced this decision.
2. The most important risk drivers, ranked by severity.
3. The business impact of proceeding without remediation.
4. Which transports or risk conditions should be remediated first.
5. What measurable conditions should be satisfied before deployment.
6. Explain the decision in a concise executive summary.

Do not invent facts that are not present in the data.
Keep the response professional and suitable for a C-suite review.
"""

            with st.spinner(
                "🤖 AI evaluating whole-dataset deployment readiness..."
            ):
                st.session_state.gate_ai_explanation = ask_ai(
                    gate_prompt
                )

        if st.session_state.get("gate_ai_explanation"):
            st.markdown("#### 📋 Executive AI Summary")
            st.info(st.session_state.gate_ai_explanation)

        # EXPORT GATE REPORT
        # --------------------------------------------------------

        st.divider()

        gate_report_df = pd.DataFrame({
            "Metric": [
                "Deployment Decision",
                "Total Transports",
                "HIGH Risk",
                "HIGH Risk %",
                "MEDIUM Risk",
                "LOW Risk",
                "Production Transports",
                "Production HIGH Risk",
                "Conflicts",
                "Historical Failures",
                "AI Anomalies"
            ],
            "Value": [
                gate_decision,
                gate_total,
                gate_high,
                f"{gate_high_pct:.1f}%",
                gate_medium,
                gate_low,
                production_count,
                production_high_count,
                conflict_count,
                failure_count,
                anomaly_count
            ]
        })

        st.markdown("### 📥 Export Deployment Gate")

        st.download_button(
            "📥 Download Deployment Gate Report",
            data=gate_report_df.to_csv(index=False),
            file_name="sap_deployment_gate_report.csv",
            mime="text/csv"
        )


        # ------------------------------------------------------------
        # AI REMEDIATION BASELINE
        # Uses the highest-risk Production transport when available.
        # ------------------------------------------------------------
        if not gate_df.empty:
            if not gate_production_high.empty:
                optimizer_row = gate_production_high.iloc[0]
            elif gate_high > 0:
                optimizer_row = gate_df[gate_df["Predicted Risk"] == "HIGH"].iloc[0]
            else:
                optimizer_row = gate_df.iloc[0]

            a_mod = optimizer_row["module"]
            a_obj = int(optimizer_row["objects_changed"])
            a_lin = int(optimizer_row["lines_changed"])
            a_con = int(optimizer_row["conflicts"])
            a_fail = int(optimizer_row["history_failures"])
            a_stg = optimizer_row["transport_stage"]
            a_stat = optimizer_row["change_request_status"]

            optimizer_baseline = pd.DataFrame([{
                "module": a_mod,
                "objects_changed": a_obj,
                "lines_changed": a_lin,
                "conflicts": a_con,
                "history_failures": a_fail,
                "transport_stage": a_stg,
                "change_request_status": a_stat
            }])

            try:
                optimizer_processed = preprocess(optimizer_baseline.copy())
                pred_a = predict(model, optimizer_processed)[0]
            except Exception:
                pred_a = optimizer_row.get("Predicted Risk", "HIGH")


        # ============================================================
        # 🔥 AI REMEDIATION OPTIMIZER
        # ============================================================
        st.divider()

        st.markdown("### 🚀 AI Remediation Optimizer")

        st.caption(
            "Automatically tests practical remediation strategies "
            "against the trained ML model and recommends the safest option."
        )

        if st.button("🧠 Find Best Remediation Strategy", key="remediation_optimizer"):

            with st.spinner("🤖 Testing remediation strategies against the ML model..."):

                # ========================================================
                # 1. CURRENT SCENARIO A = BASELINE
                # ========================================================

                baseline = {
                    "module": a_mod,
                    "objects_changed": a_obj,
                    "lines_changed": a_lin,
                    "conflicts": a_con,
                    "history_failures": a_fail,
                    "transport_stage": a_stg,
                    "change_request_status": a_stat
                }

                # ========================================================
                # 2. CANDIDATE REMEDIATION STRATEGIES
                # ========================================================

                strategies = [

                    (
                        "Resolve conflicts",
                        {
                            "conflicts": 0
                        }
                    ),

                    (
                        "Resolve historical failures",
                        {
                            "history_failures": 0
                        }
                    ),

                    (
                        "Reduce objects changed",
                        {
                            "objects_changed": max(
                                1,
                                int(a_obj * 0.5)
                            )
                        }
                    ),

                    (
                        "Reduce lines changed",
                        {
                            "lines_changed": max(
                                1,
                                int(a_lin * 0.5)
                            )
                        }
                    ),

                    (
                        "Validate in Quality before Production",
                        {
                            "transport_stage": "Quality"
                        }
                    ),

                    (
                        "Resolve conflicts + failures",
                        {
                            "conflicts": 0,
                            "history_failures": 0
                        }
                    ),

                    (
                        "Reduce changes + resolve conflicts",
                        {
                            "objects_changed": max(
                                1,
                                int(a_obj * 0.5)
                            ),
                            "lines_changed": max(
                                1,
                                int(a_lin * 0.5)
                            ),
                            "conflicts": 0
                        }
                    ),

                    (
                        "Full risk remediation",
                        {
                            "objects_changed": max(
                                1,
                                int(a_obj * 0.5)
                            ),
                            "lines_changed": max(
                                1,
                                int(a_lin * 0.5)
                            ),
                            "conflicts": 0,
                            "history_failures": 0,
                            "transport_stage": "Quality"
                        }
                    )
                ]

                # ========================================================
                # 3. RISK SCORE
                # ========================================================

                risk_score = {
                    "LOW": 1,
                    "MEDIUM": 2,
                    "HIGH": 3
                }

                results = []

                # ========================================================
                # 4. EVALUATE EVERY STRATEGY USING SAME ML MODEL
                # ========================================================

                for strategy_name, changes in strategies:

                    candidate = baseline.copy()
                    candidate.update(changes)

                    candidate_df = pd.DataFrame([candidate])

                    try:

                        # ------------------------------------------------
                        # Preprocess candidate
                        # ------------------------------------------------

                        processed = preprocess(
                            candidate_df.copy()
                        )

                        # ------------------------------------------------
                        # Predict risk
                        # ------------------------------------------------

                        candidate_pred = predict(
                            model,
                            processed
                        )[0]

                        # ------------------------------------------------
                        # Get model confidence
                        # ------------------------------------------------

                        candidate_conf, _, _ = get_confidence(
                            processed
                        )

                        if candidate_conf is not None:

                            confidence_value = round(
                                float(candidate_conf[0]),
                                2
                            )

                        else:

                            confidence_value = 0

                        # ------------------------------------------------
                        # Calculate risk score
                        # ------------------------------------------------

                        score = risk_score.get(
                            candidate_pred,
                            3
                        )

                        baseline_score = risk_score.get(
                            pred_a,
                            3
                        )

                        improvement = baseline_score - score

                        # ------------------------------------------------
                        # Store successful result
                        # ------------------------------------------------

                        results.append({
                            "Strategy": strategy_name,
                            "Predicted Risk": candidate_pred,
                            "Confidence %": confidence_value,
                            "Risk Score": score,
                            "Improvement": improvement,
                            "Error": ""
                        })

                    except Exception as e:

                        # =================================================
                        # IMPORTANT:
                        # SHOW THE REAL ERROR INSTEAD OF HIDING IT
                        # =================================================

                        results.append({
                            "Strategy": strategy_name,
                            "Predicted Risk": "ERROR",
                            "Confidence %": 0,
                            "Risk Score": 99,
                            "Improvement": -99,
                            "Error": f"{type(e).__name__}: {str(e)}"
                        })

                # ========================================================
                # 5. CREATE RESULTS DATAFRAME
                # ========================================================

                remediation_df = pd.DataFrame(results)

                # ========================================================
                # 6. SHOW DEBUG INFORMATION IF ANY STRATEGY FAILED
                # ========================================================

                error_rows = remediation_df[
                    remediation_df["Predicted Risk"] == "ERROR"
                ]

                if not error_rows.empty:

                    st.warning(
                        "⚠ Some remediation strategies could not be evaluated."
                    )

                    with st.expander("🔧 Show Remediation Debug Information"):

                        for _, error_row in error_rows.iterrows():

                            st.error(
                                f"❌ {error_row['Strategy']}\n\n"
                                f"Error: {error_row['Error']}"
                            )

                # ========================================================
                # 7. RANK RESULTS
                # ========================================================

                remediation_df = remediation_df.sort_values(
                    by=[
                        "Risk Score",
                        "Confidence %"
                    ],
                    ascending=[
                        True,
                        False
                    ]
                ).reset_index(drop=True)

                # ========================================================
                # 8. CHECK WHETHER WE HAVE ANY SUCCESSFUL RESULT
                # ========================================================

                successful_results = remediation_df[
                    remediation_df["Predicted Risk"] != "ERROR"
                ]

                if successful_results.empty:

                    st.error(
                        "❌ The model could not evaluate any remediation strategy."
                    )

                    st.info(
                        "Open 'Show Remediation Debug Information' above "
                        "to see the exact error."
                    )

                else:

                    # ====================================================
                    # 9. BEST RECOMMENDATION
                    # ====================================================

                    best = successful_results.iloc[0]

                    st.markdown(
                        "#### 🏆 Recommended Remediation"
                    )

                    # ====================================================
                    # LOW RISK
                    # ====================================================

                    if best["Predicted Risk"] == "LOW":

                        st.success(
                            f"""
        ### 🟢 LOW-RISK PATH FOUND

        **Recommended action:**  
        {best["Strategy"]}

        **Predicted Risk:** LOW

        **Model Confidence:** {best["Confidence %"]}%

        **Risk Improvement:** {best["Improvement"]} level(s)
        """
                        )

                    # ====================================================
                    # MEDIUM RISK
                    # ====================================================

                    elif best["Predicted Risk"] == "MEDIUM":

                        st.warning(
                            f"""
        ### 🟡 RISK CAN BE REDUCED

        **Recommended action:**  
        {best["Strategy"]}

        **Predicted Risk:** MEDIUM

        **Model Confidence:** {best["Confidence %"]}%

        **Risk Improvement:** {best["Improvement"]} level(s)

        Additional validation is recommended before production deployment.
        """
                        )

                    # ====================================================
                    # HIGH RISK
                    # ====================================================

                    else:

                        st.error(
                            f"""
        ### 🔴 HIGH RISK REMAINS

        **Best available action:**  
        {best["Strategy"]}

        **Predicted Risk:** HIGH

        **Model Confidence:** {best["Confidence %"]}%

        The tested remediation strategies did not reduce the transport below HIGH risk.
        """
                        )

                    # ====================================================
                    # 10. STRATEGY LEADERBOARD
                    # ====================================================

                    st.markdown(
                        "#### 📊 Strategies Tested"
                    )

                    display_df = successful_results[
                        [
                            "Strategy",
                            "Predicted Risk",
                            "Confidence %",
                            "Improvement"
                        ]
                    ]

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # ====================================================
                    # 11. VISUAL COMPARISON
                    # ====================================================

                    chart_df = successful_results.copy()

                    if not chart_df.empty:

                        fig_rem = px.bar(
                            chart_df,
                            x="Strategy",
                            y="Improvement",
                            text="Predicted Risk",
                            title="Risk Improvement by Remediation Strategy"
                        )

                        fig_rem.update_layout(
                            xaxis_tickangle=-35,
                            yaxis_title="Risk Level Improvement",
                            xaxis_title="Remediation Strategy"
                        )

                        st.plotly_chart(
                            fig_rem,
                            use_container_width=True
                        )

                    # ====================================================
                    # 12. DECISION RECOMMENDATION
                    # ====================================================

                    st.markdown(
                        "#### 🎯 Decision"
                    )

                    if best["Predicted Risk"] == "LOW":

                        st.success(
                            "✅ Recommended: Apply the remediation, "
                            "re-run validation, then proceed toward deployment."
                        )

                    elif best["Predicted Risk"] == "MEDIUM":

                        st.warning(
                            "⚠ Recommended: Apply the remediation and "
                            "perform additional testing before production."
                        )

                    else:

                        st.error(
                            "🛑 Recommended: Do not deploy directly to "
                            "Production. Further remediation is required."
                        )
                            # ============================================================
    # TAB 10 — AI RELEASE COMMANDER
    # ============================================================
    with tab10:

        st.markdown("## 🚀 AI Release Commander")

        st.caption(
            "Executive release-readiness assessment combining "
            "deployment risk, remediation status, and what-if analysis."
        )

        # --------------------------------------------------------
        # RELEASE READINESS
        # --------------------------------------------------------
        release_blockers = []

        if production_high_count > 0:
            release_blockers.append(
                f"{production_high_count:,} HIGH-risk Production transports"
            )

        if gate_high_pct >= 50:
            release_blockers.append(
                f"HIGH-risk concentration is {gate_high_pct:.1f}%"
            )

        if conflict_count > 0:
            release_blockers.append(
                f"{conflict_count:,} transport conflicts"
            )

        if failure_count > 0:
            release_blockers.append(
                f"{failure_count:,} historical failures"
            )

        # AI anomalies are monitored separately.
        # They are NOT automatic release blockers because
        # Isolation Forest intentionally flags unusual patterns.
        release_ready = len(release_blockers) == 0

        if release_ready:

            st.success(
                "🟢 RELEASE READY — No critical deployment blockers detected."
            )

        else:

            st.error(
                "🔴 RELEASE BLOCKED — Critical risk conditions require remediation."
            )

        # --------------------------------------------------------
        # RELEASE SCORE
        # --------------------------------------------------------
        risk_penalty = 0

        risk_penalty += min(gate_high_pct * 0.5, 40)

        if production_high_count > 0:
            risk_penalty += min(
                production_high_count / max(gate_total, 1) * 100,
                25
            )

        if conflict_count > 0:
            risk_penalty += min(
                conflict_count / max(gate_total, 1) * 100,
                15
            )

        if anomaly_count > 0:
            risk_penalty += min(
                anomaly_count / max(gate_total, 1) * 100,
                10
            )

        release_score = max(
            0,
            min(100, 100 - risk_penalty)
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Release Readiness",
                f"{release_score:.0f}/100"
            )

        with c2:
            st.metric(
                "Deployment Decision",
                gate_decision
            )

        with c3:
            st.metric(
                "Critical Blockers",
                len(release_blockers)
            )

        # --------------------------------------------------------
        # BLOCKERS
        # --------------------------------------------------------
        st.markdown("### 🚨 Release Blockers")

        if release_blockers:

            for i, blocker in enumerate(
                release_blockers,
                start=1
            ):
                st.warning(
                    f"**{i}.** {blocker}"
                )

        else:

            st.success(
                "No release blockers detected."
            )

        # --------------------------------------------------------
        # COMMANDER RECOMMENDATION
        # --------------------------------------------------------
        st.markdown("### 🧭 Commander Recommendation")

        if production_high_count > 0:

            st.error(
                "🛑 DO NOT RELEASE\n\n"
                "High-risk Production transports are still present. "
                "Remediate and validate them before production deployment."
            )

        elif gate_high_pct >= 50:

            st.error(
                "🛑 HOLD RELEASE\n\n"
                "The overall HIGH-risk concentration remains too high "
                "for a safe production release."
            )

        elif conflict_count > 0:

            st.warning(
                "⚠️ REVIEW BEFORE RELEASE\n\n"
                "Transport conflicts must be investigated and resolved."
            )

        else:

            st.success(
                "✅ RELEASE CAN PROCEED\n\n"
                "The current dataset does not contain critical deployment blockers."
            )

        # --------------------------------------------------------
        # EXECUTIVE SUMMARY
        # --------------------------------------------------------
        st.markdown("### 🧠 AI Executive Summary")

        summary = f"""
**Deployment Decision:** {gate_decision}

**Dataset:** {gate_total:,} transports

**HIGH Risk:** {gate_high:,} ({gate_high_pct:.1f}%)

**Production HIGH Risk:** {production_high_count:,}

**Conflicts:** {conflict_count:,}

**Historical Failures:** {failure_count:,}

**AI Anomalies:** {anomaly_count:,}

The current release assessment indicates that the deployment
should {'not proceed until the identified risks are remediated' if not release_ready else 'proceed subject to normal release controls'}.
"""

        st.info(summary)

        # --------------------------------------------------------
        # RELEASE CHECKLIST
        # --------------------------------------------------------
        st.markdown("### ✅ Release Checklist")

        checklist = [
            (
                "HIGH-risk Production transports resolved",
                production_high_count == 0
            ),
            (
                "Overall HIGH-risk concentration acceptable",
                gate_high_pct < 50
            ),
            (
                "Transport conflicts resolved",
                conflict_count == 0
            ),
            (
                "Historical failure conditions reviewed",
                failure_count == 0
            ),
            (
                f"AI anomaly monitoring completed ({anomaly_count:,} anomalies detected)",
                True
            )
        ]

        for item, passed in checklist:

            if passed:
                st.success(f"✅ {item}")

            else:
                st.error(f"❌ {item}")

        # --------------------------------------------------------
        # FINAL COMMAND
        # --------------------------------------------------------
        st.divider()

        if release_ready:

            st.markdown(
                """
                <div style="
                    padding:25px;
                    border-radius:15px;
                    background:#123d2b;
                    border:2px solid #2ecc71;
                    text-align:center;
                ">
                    <div style="font-size:3rem;">🟢</div>
                    <div style="
                        font-size:2rem;
                        font-weight:bold;
                        color:#2ecc71;
                    ">
                        RELEASE APPROVED
                    </div>
                    <div style="
                        color:#ddd;
                        margin-top:10px;
                    ">
                        AI Release Commander recommends proceeding.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div style="
                    padding:25px;
                    border-radius:15px;
                    background:#451f25;
                    border:2px solid #ff4d4d;
                    text-align:center;
                ">
                    <div style="font-size:3rem;">🔴</div>
                    <div style="
                        font-size:2rem;
                        font-weight:bold;
                        color:#ff4d4d;
                    ">
                        RELEASE BLOCKED
                    </div>
                    <div style="
                        color:#ddd;
                        margin-top:10px;
                    ">
                        Remediate the identified risks and re-run
                        the Deployment Gate before release.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
                    # --------------------------------------------------------
        # HUMAN RELEASE DECISION
        # --------------------------------------------------------
        st.divider()

        st.markdown("### 👤 Human Release Decision")

        st.caption(
            "AI provides the recommendation. The authorized user "
            "retains final decision authority."
        )

        # Current AI recommendation
        if release_ready:
            ai_recommendation = "GO"
        else:
            ai_recommendation = "NO-GO"

        st.info(
            f"🤖 **AI Recommendation:** {ai_recommendation}  \n"
            f"📊 **Release Readiness Score:** {release_score:.0f}/100"
        )

        human_decision = st.radio(
            "Select your final decision",
            [
                "Accept AI Recommendation",
                "Override AI Recommendation"
            ],
            key="release_human_decision"
        )

        # --------------------------------------------------------
        # ACCEPT AI RECOMMENDATION
        # --------------------------------------------------------
        if human_decision == "Accept AI Recommendation":

            final_decision = ai_recommendation

            if final_decision == "GO":

                st.success(
                    "🟢 Final Decision: GO — AI recommendation accepted."
                )

            else:

                st.error(
                    "🔴 Final Decision: NO-GO — AI recommendation accepted."
                )

        # --------------------------------------------------------
        # OVERRIDE AI RECOMMENDATION
        # --------------------------------------------------------
        else:

            st.warning(
                f"⚠️ You are overriding the AI recommendation: "
                f"**{ai_recommendation}**"
            )

            override_decision = st.radio(
                "Select the final decision",
                ["GO", "NO-GO"],
                horizontal=True,
                key="override_final_decision"
            )

            override_reason = st.text_area(
                "Reason for overriding the AI recommendation",
                placeholder=(
                    "Enter the business or operational reason "
                    "for overriding the AI recommendation..."
                ),
                key="override_reason"
            )

            if st.button(
                "💾 Confirm Human Override",
                type="primary",
                use_container_width=True,
                key="confirm_release_override"
            ):

                if not override_reason.strip():

                    st.error(
                        "Please provide a reason for the human override."
                    )

                elif override_decision == ai_recommendation:

                    st.warning(
                        "The selected decision is the same as the AI "
                        "recommendation. Please select a different decision "
                        "to perform an override."
                    )

                else:

                    user_email = st.session_state.get(
                        "user_email",
                        "Unknown User"
                    )

                    try:

                        save_release_override(
                            transport_id="RELEASE_GATE",
                            ai_recommendation=ai_recommendation,
                            ai_risk_score=release_score,
                            final_decision=override_decision,
                            override_reason=override_reason.strip(),
                            user_email=user_email
                        )

                        st.success(
                            f"✅ Human override recorded successfully. "
                            f"Final Decision: {override_decision}"
                        )

                        st.info(
                            f"👤 Decision by: {user_email}"
                        )

                    except Exception as e:

                        st.error(
                            f"Unable to save the override decision: {e}"
                        )
                    
                     # --------------------------------------------------------
    # EMPTY STATE
    # ================================================================
else:
    st.markdown("""
    <div style='text-align:center;padding:60px 20px;
         background:linear-gradient(135deg,#1a1a2e,#16213e);
         border-radius:20px;border:1px dashed #764ba2;margin-top:20px'>
        <div style='font-size:4em'>📂</div>
        <h2 style='color:white;margin-top:15px'>
            Upload CSV or Load from Firebase
        </h2>
        <p style='color:#aaa;font-size:1.1em'>
            Choose a data source from the sidebar to begin
        </p>
        <div style='margin-top:20px;display:flex;
             justify-content:center;gap:30px;flex-wrap:wrap'>
            <div style='color:#764ba2'>
                <div style='font-size:2em'>📁</div>
                <div>Upload CSV</div>
            </div>
            <div style='color:#4caf50'>
                <div style='font-size:2em'>🔥</div>
                <div>Load from Firebase</div>
            </div>
        </div>
        <p style='color:#555;margin-top:25px;font-size:0.9em'>
            Required columns: transport_id, module, transport_stage,
            objects_changed, lines_changed, conflicts,
            history_failures, change_request_status
        </p>
    </div>
    """, unsafe_allow_html=True)
