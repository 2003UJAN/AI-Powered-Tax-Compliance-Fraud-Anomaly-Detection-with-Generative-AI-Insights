# AI-Powered Tax Fraud & Anomaly Detection

This project uses an Isolation Forest model to detect anomalous financial transactions and the Google Gemini API to provide human-readable explanations for why a transaction is flagged.

## 📂 Repository Structure

- `data/`: Contains the transaction dataset.
- `model/`: Contains the trained machine learning model and feature list.
- `notebooks/`: Jupyter/Colab notebook for model development.
- `app.py`: The main Streamlit dashboard application.
- `generate_data.py`: Script to create the `complex_transactions.csv` dataset.
- `requirements.txt`: Project dependencies.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd tax-fraud-detector
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    -   Copy `.env.example` to a new file named `.env`.
    -   Add your Google Gemini API key to the `.env` file.

4.  **Generate data and train model:**
    -   Run the data generation script: `python generate_data.py`
    -   Run the `notebooks/01_model_training.ipynb` notebook to create the model files.

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
