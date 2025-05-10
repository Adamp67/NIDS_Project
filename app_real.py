import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NIDS Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; }
    .info-box { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; }
    .metric-box { background-color: #ffffff; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Network Intrusion Detection Demo")

with st.sidebar:
    st.header("📚 Dataset Information")
    dataset_choice = st.selectbox("Select Dataset", ["NSL-KDD", "UNSW-NB15"])
    if dataset_choice == "NSL-KDD":
        with st.expander("NSL-KDD Dataset"):
            st.markdown("""
            **NSL-KDD Dataset Features:**
            - 41 features including:
              - Basic features (duration, protocol_type, service)
              - Content features (src_bytes, dst_bytes)
              - Time-based features (count, srv_count)
              - Host-based features (dst_host_count, dst_host_srv_count)
            **Attack Types:**
            - DoS, Probe, R2L, U2R
            **Common Features:**
            - protocol_type: TCP, UDP, ICMP
            - service: http, ftp, smtp, etc.
            - flag: connection status
            """)
    else:
        with st.expander("UNSW-NB15 Dataset"):
            st.markdown("""
            **UNSW-NB15 Dataset Features:**
            - 49 features including:
              - Basic features (dur, proto, service)
              - Statistical features (sbytes, dbytes)
              - Time-based features (sttl, dttl)
              - Connection features (state, sload, dload)
            **Attack Categories:**
            - Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms
            **Key Features:**
            - proto: protocol type
            - service: network service
            - state: connection state
            """)

st.markdown("""
This application uses machine learning to detect network intrusions and attacks.\
Select your dataset, then upload your network traffic data in CSV or Parquet format to analyze it.
""")

# NSL-KDD columns
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]
DEFAULT_VALUES = {
    'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0, 'root_shell': 0,
    'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0
}

# UNSW-NB15 columns (minimal for demo, ideally load from unsw_feature_names.pkl)
UNSW_CAT_COLS = ['proto', 'service', 'state']

# Model/scaler loader
@st.cache_resource
def load_model_and_scaler(dataset):
    try:
        if dataset == "NSL-KDD":
            model = joblib.load(Path("NIDS_Data/NSL_KDD/best_rf_model.pkl"))
            scaler = joblib.load(Path("NIDS_Data/NSL_KDD/nslkdd_scaler.pkl"))
        else:
            model = joblib.load(Path("NIDS_Data/UNSW_NB15/final_rf_model.pkl"))
            scaler = joblib.load(Path("NIDS_Data/UNSW_NB15/unsw_scaler.pkl"))
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model/scaler: {e}")
        return None, None

# Preprocessing
def preprocess_nslkdd(df, scaler):
    try:
        df.columns = df.columns.str.lower()
        missing_cols = set(NSL_KDD_COLUMNS) - set(df.columns)
        if missing_cols:
            st.warning(f"Missing columns will be filled with default values: {', '.join(missing_cols)}")
            for col in missing_cols:
                df[col] = DEFAULT_VALUES.get(col, 0)
        df = df[NSL_KDD_COLUMNS]
        for col in ['protocol_type', 'service', 'flag']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        return scaler.transform(df)
    except Exception as e:
        st.error(f"❌ Error during NSL-KDD preprocessing: {e}")
        return None

def preprocess_unsw(df, scaler):
    try:
        # Drop label columns not used in prediction
        for drop_col in ['attack_cat', 'label']:
            if drop_col in df.columns:
                df.drop(columns=[drop_col], inplace=True)

        # Safely encode categorical columns
        for col in UNSW_CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)
                df[col] = pd.factorize(df[col])[0]

        # Drop any remaining categorical dtype columns
        for col in df.select_dtypes(include='category').columns:
            df[col] = df[col].astype(str)

        # Fill any missing values
        df = df.fillna(0)

        # Scale numerical features
        return scaler.transform(df)

    except Exception as e:
        st.error(f"❌ Error during UNSW-NB15 preprocessing: {e}")
        return None


# Prediction and visualization
def predict_and_plot(df, model, scaler, dataset):
    try:
        if dataset == "NSL-KDD":
            X_scaled = preprocess_nslkdd(df.copy(), scaler)
        else:
            X_scaled = preprocess_unsw(df.copy(), scaler)
        if X_scaled is None:
            return
        preds = model.predict(X_scaled)
        proba = model.predict_proba(X_scaled)[:, 1]
        results = pd.DataFrame({
            'Prediction': np.where(preds == 0, "Benign", "Attack"),
            'Attack Probability': proba
        })
        st.subheader("📊 Prediction Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(results))
        with col2:
            st.metric("Benign Connections", sum(results['Prediction'] == 'Benign'))
        with col3:
            st.metric("Attack Connections", sum(results['Prediction'] == 'Attack'))
        st.subheader("📈 Prediction Distribution")
        fig = px.pie(results, names='Prediction', title='Prediction Distribution', color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig)
        st.subheader("📊 Attack Probability Distribution")
        fig_prob = px.histogram(results, x='Attack Probability', color='Prediction', title='Attack Probability Distribution', color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig_prob)
        st.subheader("🎯 Model Performance")
        cm = confusion_matrix(preds, preds)
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Benign', 'Attack'], y=['Benign', 'Attack'], colorscale='Blues'))
        fig_cm.update_layout(title="Confusion Matrix")
        st.plotly_chart(fig_cm)
        st.subheader("📋 Detailed Results")
        st.dataframe(results)
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# Main app flow
st.header("📤 Upload Your Network Traffic Data")
if dataset_choice == "NSL-KDD":
    uploaded_file = st.file_uploader("Choose a CSV file (NSL-KDD)", type=["csv"])
else:
    uploaded_file = st.file_uploader("Choose a CSV or Parquet file (UNSW-NB15)", type=["csv", "parquet"])

if uploaded_file is not None:
    try:
        if dataset_choice == "NSL-KDD":
            df = pd.read_csv(uploaded_file)
        else:
            if uploaded_file.name.endswith(".parquet"):
                df = pd.read_parquet(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        st.subheader("📈 Data Statistics")
        st.dataframe(df.describe())
        if st.button("🔍 Analyze Traffic", type="primary"):
            with st.spinner("Processing..."):
                model, scaler = load_model_and_scaler(dataset_choice)
                if model is not None and scaler is not None:
                    predict_and_plot(df, model, scaler, dataset_choice)
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

st.markdown("---")
st.caption("Developed for Computer Science Project by Adam Patel")