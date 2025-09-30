import streamlit as st
import pandas as pd
import joblib
import time
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- 1. Page Configuration & Styling ---
st.set_page_config(
    page_title="Live Fraud Detection Engine",
    page_icon="🛡️",
    layout="wide"
)

# --- 2. Gemini API Configuration ---
api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
gen_model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Failed to configure Gemini API: {e}")
else:
    st.warning("🔑 Gemini API Key not found. AI analysis will be disabled.", icon="⚠️")

# --- 3. Function to Generate Insights ---
def get_gemini_explanation(transaction_data):
    if not gen_model:
        return "AI analysis is disabled due to missing API key."

    prompt = f"""
    Analyze the financial transaction flagged for potential tax fraud/anomaly.
    Provide a concise, expert analysis (3-4 sentences) explaining the suspicious indicators.

    Transaction Data:
    - Amount: {transaction_data.get('Amount')} {transaction_data.get('Currency')}
    - Sender: {transaction_data.get('Sender_Name')}
    - Sender's Historical Avg. Amount: {transaction_data.get('Avg_Transaction_Value_Last_30D')}
    - Sender/Recipient Location: From {transaction_data.get('Sender_Country')} to {transaction_data.get('Recipient_Country')}
    - Sender's 24H Activity: {transaction_data.get('Transactions_Last_24H')} transactions
    - New Recipient?: {'Yes' if transaction_data.get('Is_New_Recipient') == 1 else 'No'}

    Identify the most likely fraud pattern: Is this indicative of structuring (smurfing), an unusually high-value transfer, or a geographical red flag?
    """
    try:
        response = gen_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error contacting Gemini API: {e}"

# --- 4. Load Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load('model/isolation_forest_model_complex.pkl')
    df = pd.read_csv('data/complex_transactions.csv', parse_dates=['Timestamp'])
    features = joblib.load('model/feature_list.pkl')
    return model, df, features

try:
    model, df, feature_list = load_assets()
    assets_loaded = True
    # Get unique values for dropdowns
    CURRENCIES = df['Currency'].unique()
    COUNTRIES = sorted(df['Sender_Country'].unique())
    TRANSACTION_TYPES = df['Transaction_Type'].unique()
    DEVICE_TYPES = df['Device_Type'].unique()
except FileNotFoundError:
    st.error("Error: Model or data files not found. Please run the data generation and model training scripts first.")
    assets_loaded = False

# --- 5. Manual Entry & Prediction Logic (Sidebar) ---
if assets_loaded:
    with st.sidebar:
        st.header("Manual Anomaly Check")
        with st.form("manual_entry_form"):
            st.write("Enter transaction details to check if it's an anomaly.")

            # Form inputs
            amount = st.number_input("Amount", min_value=0.0, step=10.0, format="%.2f")
            currency = st.selectbox("Currency", options=CURRENCIES)
            sender_country = st.selectbox("Sender Country", options=COUNTRIES)
            recipient_country = st.selectbox("Recipient Country", options=COUNTRIES)
            transactions_last_24h = st.number_input("Sender's Transactions (Last 24H)", min_value=0, step=1)
            avg_transaction_value = st.number_input("Sender's Avg. Transaction Value (Last 30D)", min_value=0.0, step=10.0, format="%.2f")
            is_new_recipient_option = st.selectbox("Is this a new recipient for the sender?", options=["No", "Yes"])
            
            # Dummy fields for schema consistency
            transaction_type = st.selectbox("Transaction Type", options=TRANSACTION_TYPES)
            device_type = st.selectbox("Device Type", options=DEVICE_TYPES)
            sender_name = st.text_input("Sender Name (Optional)", "John Doe")
            recipient_name = st.text_input("Recipient Name (Optional)", "Jane Smith")

            submit_button = st.form_submit_button(label="Check for Anomaly")

        if submit_button:
            # 1. Create a DataFrame from user input
            is_new_recipient = 1 if is_new_recipient_option == "Yes" else 0
            user_data = {
                'Amount': [amount], 'Currency': [currency], 'Sender_Name': [sender_name], 'Recipient_Name': [recipient_name],
                'Sender_Country': [sender_country], 'Recipient_Country': [recipient_country],
                'Transaction_Type': [transaction_type], 'Device_Type': [device_type],
                'Transactions_Last_24H': [transactions_last_24h], 'Avg_Transaction_Value_Last_30D': [avg_transaction_value],
                'Is_New_Recipient': [is_new_recipient]
            }
            input_df = pd.DataFrame(user_data)

            # 2. Preprocess the input data exactly as the training data
            input_encoded = pd.get_dummies(input_df, columns=['Currency', 'Transaction_Type', 'Device_Type', 'Sender_Country', 'Recipient_Country'])
            input_aligned = input_encoded.reindex(columns=feature_list, fill_value=0)

            # 3. Make a prediction
            prediction = model.predict(input_aligned)
            
            # 4. Display the result
            st.subheader("Analysis Result")
            if prediction[0] == -1:
                st.error("🚨 Anomaly Detected!", icon="🚨")
                with st.spinner("Gemini is analyzing..."):
                    explanation = get_gemini_explanation(user_data)
                    st.info(explanation, icon="🧠")
            else:
                st.success("✅ This transaction appears to be normal.", icon="✅")


# --- 6. Main Page UI ---
st.title("🛡️ Real-Time Fraud & Anomaly Detection Engine")
st.caption(f"Scanning a dataset of {len(df):,} transactions using an Isolation Forest model and Gemini 1.5 Pro for analysis.")

if assets_loaded:
    # --- Summary Metrics ---
    st.subheader("Dataset Overview", divider='rainbow')
    anomalies_count = (df['Is_Anomaly'] == 1).sum() # Assuming pre-labeled anomalies for display
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Pre-labeled Anomalies", f"{anomalies_count:,}", delta=f"{(anomalies_count/len(df))*100:.2f}% of total")
    col3.metric("Countries Monitored", f"{df['Sender_Country'].nunique()}")

    st.subheader("Live Anomaly Feed (Simulation)", divider='rainbow')
    if st.button("▶️ Start Real-Time Simulation", type="primary"):
        df_encoded = pd.get_dummies(df, columns=['Currency', 'Transaction_Type', 'Device_Type', 'Sender_Country', 'Recipient_Country'])
        df_aligned = df_encoded.reindex(columns=feature_list, fill_value=0)
        df['anomaly_score'] = model.predict(df_aligned)
        anomalies = df[df['anomaly_score'] == -1].sort_values(by='Timestamp', ascending=False)
        
        if anomalies.empty:
            st.success("✅ Scanned the entire dataset. No anomalies were found!")
        else:
            progress_text = "Simulation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            st.info(f"Initiating simulation... found {len(anomalies)} anomalies to display.")
            time.sleep(2)
            
            placeholder = st.empty()
            
            for i, (index, row) in enumerate(anomalies.iterrows()):
                progress_percent = (i + 1) / len(anomalies)
                my_bar.progress(progress_percent, text=f"Displaying anomaly {i+1} of {len(anomalies)}")

                with placeholder.container():
                    with st.container(border=True):
                        col_a, col_b = st.columns([2, 3])
                        with col_a:
                            st.markdown(f"##### 🚨 Anomaly Detected")
                            st.write(f"**Timestamp:** `{row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}`")
                            st.markdown(f"**Amount:** <span style='color:red; font-size: 20px; font-weight: bold;'>{row['Amount']:.2f} {row['Currency']}</span>", unsafe_allow_html=True)
                            st.write(f"**Sender:** 👤 {row['Sender_Name']}")
                            st.write(f"**Recipient:** 👤 {row['Recipient_Name']}")
                            st.write(f"**Route:** 🌍 {row['Sender_Country']} → {row['Recipient_Country']}")
                        with col_b:
                            st.markdown("##### 🤖 Gemini 2.0 Flash Analysis")
                            with st.spinner("Gemini is analyzing..."):
                                explanation = get_gemini_explanation(row.to_dict())
                                st.info(explanation, icon="🧠")
                    time.sleep(3)
            
            my_bar.progress(1.0, text="Simulation Complete!")
            st.success("✅ Simulation finished. All flagged anomalies have been processed.")
