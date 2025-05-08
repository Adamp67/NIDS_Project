# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import shap
import warnings
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🛡️ Network Intrusion Detection System (NIDS)")
st.markdown("""
This application uses a hybrid approach of classification and anomaly detection models 
to predict whether a network connection is **Benign (Safe)** or **Malicious (Attack)**.
""")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This NIDS application combines:
    - Random Forest Classification
    - Isolation Forest Anomaly Detection
    - Deep Autoencoder Anomaly Detection
    - SHAP Explainability
    """)
    
    st.markdown("---")
    st.header("📚 Information")
    with st.expander("Dataset Information"):
        st.markdown("""
        **NSL-KDD Dataset:**
        - Contains 41 features
        - Includes both normal and attack traffic
        - Common features: duration, protocol_type, service, flag, etc.
        """)
    
    st.markdown("---")
    st.caption("Developed for Computer Science Project by Adam Patel")

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix using plotly."""
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Benign', 'Predicted Attack'],
        y=['Actual Benign', 'Actual Attack'],
        colorscale='Blues',
        showscale=True
    ))
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label'
    )
    return fig

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve using plotly."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'ROC curve (AUC = {roc_auc:.2f})',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random',
        line=dict(color='gray', dash='dash')
    ))
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance using plotly."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[feature_names[i] for i in indices[:15]],
        y=importances[indices[:15]],
        marker_color='rgb(55, 83, 109)'
    ))
    fig.update_layout(
        title='Top 15 Feature Importances',
        xaxis_title='Features',
        yaxis_title='Importance Score',
        showlegend=False
    )
    return fig

def plot_shap_summary(model, X_sample, feature_names):
    """Plot SHAP summary using plotly."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification
    
    # Get mean absolute SHAP values for each feature
    mean_shap = np.abs(shap_values).mean(0)
    indices = np.argsort(mean_shap)[::-1]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[feature_names[i] for i in indices[:15]],
        y=mean_shap[indices[:15]],
        marker_color='rgb(55, 83, 109)'
    ))
    fig.update_layout(
        title='SHAP Feature Importance',
        xaxis_title='Features',
        yaxis_title='mean |SHAP value|',
        showlegend=False
    )
    return fig

# Load models and scalers
@st.cache_resource
def load_models():
    try:
        data_dir = Path("NIDS_Data")
        hybrid_path = data_dir / "hybrid"
        
        # Load models with error handling
        try:
            rf_model = joblib.load(data_dir / 'final_rf_model.pkl')
        except Exception as e:
            st.error(f"❌ Error loading Random Forest model: {str(e)}")
            return None, None, None, None, None
            
        try:
            iso_model = joblib.load(data_dir / 'final_shap_explainer.pkl')
        except Exception as e:
            st.error(f"❌ Error loading Isolation Forest model: {str(e)}")
            return None, None, None, None, None
            
        try:
            scaler = joblib.load(data_dir / 'final_scaler.pkl')
        except Exception as e:
            st.error(f"❌ Error loading scaler: {str(e)}")
            return None, None, None, None, None
            
        try:
            fusion_model = joblib.load(hybrid_path / 'fusion_meta_model.pkl')
        except Exception as e:
            st.error(f"❌ Error loading fusion model: {str(e)}")
            return None, None, None, None, None
            
        try:
            autoencoder_model = load_model(hybrid_path / 'deep_autoencoder_model.h5', compile=False)
            with open(hybrid_path / 'anomaly_threshold.txt', 'r') as f:
                anomaly_threshold = float(f.read())
        except Exception as e:
            st.error(f"❌ Error loading autoencoder model: {str(e)}")
            return None, None, None, None, None
            
        return rf_model, iso_model, scaler, fusion_model, (autoencoder_model, anomaly_threshold)
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None

# Load models
rf_model, iso_model, scaler, fusion_model, (autoencoder_model, anomaly_threshold) = load_models()

# NSL-KDD column definitions
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

# Default values for commonly missing columns
DEFAULT_VALUES = {
    'num_outbound_cmds': 0,
    'is_host_login': 0,
    'is_guest_login': 0,
    'root_shell': 0,
    'su_attempted': 0,
    'num_root': 0,
    'num_file_creations': 0,
    'num_shells': 0,
    'num_access_files': 0
}

def preprocess_data(df):
    """Preprocess the NSL-KDD data with enhanced error handling and validation."""
    try:
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Find missing columns
        missing_cols = set(NSL_KDD_COLUMNS) - set(df.columns)
        
        if missing_cols:
            st.warning(f"""
            ⚠️ The following columns are missing and will be filled with default values:
            {', '.join(missing_cols)}
            
            This might affect prediction accuracy. Please ensure your dataset contains all required columns for best results.
            """)
            
            # Add missing columns with default values
            for col in missing_cols:
                if col in DEFAULT_VALUES:
                    df[col] = DEFAULT_VALUES[col]
                else:
                    df[col] = 0  # Default to 0 for numeric columns
        
        # Ensure all required columns are present
        df = df[NSL_KDD_COLUMNS]
        
        # Convert categorical columns to numerical
        categorical_cols = ['protocol_type', 'service', 'flag']
        label_encoders = {}
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Convert all columns to numeric
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                st.error(f"❌ Error converting column {col} to numeric: {str(e)}")
                return None
        
        # Fill any remaining NaN values with 0
        df = df.fillna(0)
        
        # Validate data types
        if not all(df.dtypes == 'float64' or df.dtypes == 'int64'):
            st.error("❌ Some columns could not be converted to numeric types")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error during preprocessing: {str(e)}")
        return None

def hybrid_predict(df):
    """Make predictions using the hybrid approach with enhanced error handling."""
    try:
        # Scale features
        X_scaled = scaler.transform(df)
        
        # Get predictions from each model
        rf_preds = rf_model.predict(X_scaled)
        rf_proba = rf_model.predict_proba(X_scaled)[:, 1]
        
        iso_scores = iso_model.decision_function(X_scaled)
        iso_preds = iso_model.predict(X_scaled)
        
        # Get fusion model predictions
        fusion_preds = fusion_model.predict(X_scaled)
        fusion_proba = fusion_model.predict_proba(X_scaled)[:, 1]
        
        # Get autoencoder predictions
        recon = autoencoder_model.predict(X_scaled)
        recon_errors = np.mean(np.square(X_scaled - recon), axis=1)
        auto_preds = (recon_errors > anomaly_threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'RF Classification': ['Benign' if label == 0 else 'Attack' for label in rf_preds],
            'RF Probability': rf_proba,
            'Isolation Forest': ['Normal' if anomaly == 1 else 'Anomaly' for anomaly in iso_preds],
            'Anomaly Score': iso_scores,
            'Fusion Model': ['Benign' if label == 0 else 'Attack' for label in fusion_preds],
            'Fusion Probability': fusion_proba,
            'Autoencoder': ['Benign' if pred == 0 else 'Attack' for pred in auto_preds],
            'Reconstruction Error': recon_errors
        })
        
        # Add final hybrid decision
        results['Final Decision'] = results.apply(
            lambda row: 'Attack' if (
                row['RF Classification'] == 'Attack' or
                row['Isolation Forest'] == 'Anomaly' or
                row['Fusion Model'] == 'Attack' or
                row['Autoencoder'] == 'Attack'
            ) else 'Benign',
            axis=1
        )
        
        return results, rf_preds, rf_proba, fusion_preds, fusion_proba
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None, None, None

# Main content
st.header("📤 Upload Your NSL-KDD Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        input_df = pd.read_csv(uploaded_file)
        
        # Display data preview
        st.subheader("📊 Data Preview")
        st.dataframe(input_df.head())
        
        # Show data statistics
        st.subheader("📈 Data Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(input_df.describe())
        with col2:
            # Plot feature distributions
            numeric_cols = input_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select feature to visualize", numeric_cols)
                fig = px.histogram(input_df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig)
        
        if st.button("🔍 Analyze and Predict", type="primary"):
            with st.spinner("Processing..."):
                # Preprocess data
                processed_df = preprocess_data(input_df)
                
                if processed_df is not None:
                    # Make predictions
                    results, rf_preds, rf_proba, fusion_preds, fusion_proba = hybrid_predict(processed_df)
                    
                    if results is not None:
                        st.success("✅ Analysis Complete!")
                        
                        # Display results
                        st.subheader("📊 Prediction Results")
                        st.dataframe(results)
                        
                        # Show summary statistics
                        st.subheader("📈 Summary Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Records", len(results))
                        with col2:
                            st.metric("Benign Connections", sum(results['Final Decision'] == 'Benign'))
                        with col3:
                            st.metric("Attack Connections", sum(results['Final Decision'] == 'Attack'))
                        with col4:
                            st.metric("Anomalies Detected", sum(results['Isolation Forest'] == 'Anomaly'))
                        
                        # Visualizations
                        st.subheader("📊 Visualizations")
                        
                        # Final Decision Distribution
                        fig1 = px.pie(
                            results,
                            names='Final Decision',
                            title='Final Decision Distribution',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig1)
                        
                        # Model Agreement Analysis
                        st.subheader("🎯 Model Agreement Analysis")
                        
                        # Create agreement matrix
                        agreement_matrix = pd.crosstab(
                            results['RF Classification'],
                            results['Fusion Model'],
                            margins=True
                        )
                        st.dataframe(agreement_matrix)
                        
                        # Model Performance Metrics
                        st.subheader("🎯 Model Performance")
                        
                        # Confusion Matrix for RF
                        st.plotly_chart(plot_confusion_matrix(
                            [0 if x == 'Benign' else 1 for x in results['RF Classification']],
                            rf_preds
                        ))
                        
                        # ROC Curve for RF
                        st.plotly_chart(plot_roc_curve(
                            [0 if x == 'Benign' else 1 for x in results['RF Classification']],
                            rf_proba
                        ))
                        
                        # Feature Importance
                        st.plotly_chart(plot_feature_importance(rf_model, NSL_KDD_COLUMNS))
                        
                        # SHAP Summary
                        st.plotly_chart(plot_shap_summary(rf_model, processed_df.values, NSL_KDD_COLUMNS))
                        
                        # Reconstruction Error Distribution
                        fig2 = px.histogram(
                            results,
                            x='Reconstruction Error',
                            color='Final Decision',
                            title='Reconstruction Error Distribution by Decision',
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig2)
                        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.caption("Developed for Computer Science Project by Adam Patel")