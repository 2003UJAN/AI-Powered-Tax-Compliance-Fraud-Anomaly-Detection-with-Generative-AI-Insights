import streamlit as st
import pandas as pd
import joblib
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# --- 1. Page & Gemini API Configuration ---
st.set_page_config(page_title="Real-Time Tax Fraud Detection", page_icon="🚨", layout="wide")

# Configure Gemini API
# For local dev, it uses .env. For deployment, it uses Streamlit secrets.
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
gen_model = None
if api_key:
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel('gemini-pro')
else:
    st.warning("🔑 Gemini API Key not found. Please set it in your .env file or Streamlit secrets.", icon="⚠️")


# --- 2. Function to Generate Insights via Gemini ---
def get_gemini_explanation(transaction_data):
    if not gen_model:
        return "Generative AI model is not configured."
    
    prompt = f"""
    Analyze the financial transaction flagged for potential tax fraud/anomaly.
    Provide a concise, expert analysis (3-4 sentences) explaining the suspicious indicators.
    Transaction Data:
    - Amount: {transaction_data['Amount']} {transaction_data['Currency']}
    - Sender's Historical Avg. Amount: {transaction_data['Avg_Transaction_Value_Last_30D']}
    - Transaction Type: {transaction_data['Transaction_Type']}
    - Sender/Recipient Location: {transaction_data['Sender_Country_Code']} -> {transaction_data['Recipient_Country_Code']}
    - Device: {transaction_data['Device_Type']}
    - Sender's 24H Activity: {transaction_data['Transactions_Last_24H']} transactions
    - New Recipient?: {'Yes' if transaction_data['Is_New_Recipient'] == 1 else 'No'}
    Identify the pattern: Is this structuring (smurfing), an unusually high-value transfer, high-frequency automation, or a geographical red flag?
    """
    try:
        response = gen_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error contacting Gemini API: {e}"

# --- 3. Load Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load('model/isolation_forest_model_complex.pkl')
    df = pd.read_csv('data/complex_transactions.csv')
    features = joblib.load('model/feature_list.pkl')
    return model, df, features

try:
    model, df, feature_list = load_assets()
    assets_loaded = True
except FileNotFoundError:
    st.error("Error: Model or data files not found. Please run the data generation and model training scripts first.")
    assets_loaded = False

# --- 4. Streamlit UI ---
st.title("🚨 AI-Powered Tax Fraud & Anomaly Detection")
st.write("This dashboard simulates a transaction feed, flags anomalies, and uses the **Google Gemini API** for analysis.")

if assets_loaded:
    df_encoded = pd.get_dummies(df, columns=['Currency', 'Transaction_Type', 'Device_Type', 'Sender_Country_Code'])
    df_aligned = df_encoded.reindex(columns=feature_list, fill_value=0)

    df['anomaly_score'] = model.predict(df_aligned)
    anomalies = df[df['anomaly_score'] == -1].sort_values(by='Timestamp', ascending=False)

    st.header("Live Anomaly Feed")
    if st.button("Start Real-Time Simulation", type="primary"):
        if anomalies.empty:
            st.success("Scanned the dataset. No anomalies were found!")
        else:
            placeholder = st.empty()
            for index, row in anomalies.iterrows():
                with placeholder.container():
                    st.warning(f"**Anomaly Detected!** Transaction from Sender: `{row['Sender_ID']}`")
                    col1, col2 = st.columns([1, 1.5])
                    with col1:
                        st.subheader("Transaction Details")
                        st.json(row.drop(['anomaly_score']).to_dict())
                    with col2:
                        st.subheader("🤖 Gemini API Analysis")
                        with st.spinner("Gemini is analyzing..."):
                            explanation = get_gemini_explanation(row)
                            st.info(explanation)
                    st.markdown("---")
                    time.sleep(4)
            st.success("Simulation finished.")
