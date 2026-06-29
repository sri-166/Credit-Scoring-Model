import os

import joblib
import pandas as pd
import streamlit as st

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Credit Scoring Model",
    layout="wide",
)


st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #0B2545 0%, #134074 100%);
    padding: 1.8rem 2rem;
    border-radius: 10px;
    margin-bottom: 1.5rem;
}
.main-header h1 { color: #FFFFFF; margin-bottom: 0.3rem; }
.main-header p { color: #CDE5F5; margin: 0; font-size: 1.05rem; }

.result-box-low {
    background-color: #E8F5E9;
    border-left: 6px solid #2E7D32;
    padding: 1.3rem 1.6rem;
    border-radius: 8px;
}
.result-box-high {
    background-color: #FDECEA;
    border-left: 6px solid #C62828;
    padding: 1.3rem 1.6rem;
    border-radius: 8px;
}
.result-box-low h2, .result-box-high h2 { margin: 0 0 0.4rem 0; }
</style>
""", unsafe_allow_html=True)


# LOAD THE SAVED MODEL AND PREPROCESSING OBJECTS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")


@st.cache_resource
def load_artifacts():
    """Loads the trained model and the preprocessing objects it needs.
    Cached so the (slightly slow) disk loading only happens once."""
    model = joblib.load(os.path.join(MODELS_DIR, "best_credit_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    payment_encoder = joblib.load(os.path.join(MODELS_DIR, "payment_history_encoder.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    return model, scaler, payment_encoder, feature_columns


def load_model_info():
    """Loads the best model's name and metrics table, if available.
    These are just informational and the app still works without them."""
    model_name, metrics_df = None, None

    name_path = os.path.join(MODELS_DIR, "best_model_name.txt")
    if os.path.exists(name_path):
        with open(name_path) as f:
            model_name = f.read().strip()

    metrics_path = os.path.join(MODELS_DIR, "model_metrics.csv")
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path, index_col=0)

    return model_name, metrics_df


try:
    model, scaler, payment_encoder, feature_columns = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

model_name, metrics_df = load_model_info()

# HEADER
st.markdown("""
<div class="main-header">
    <h1> Credit Scoring Model</h1>
    <p>A simple machine learning app that predicts whether a customer is a Low Risk or High Risk credit customer.</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(
        "Model files were not found in the `models/` folder. "
        "Please run `python train_model.py` from the project's root folder first, "
        "then refresh this page."
    )
    st.stop()

# SIDEBAR - PROJECT & MODEL INFO
with st.sidebar:
    st.header("ℹ️ About This App")
    st.write(
        "This tool uses historical credit data and a trained classification "
        "model to estimate credit risk for a new customer, based on their "
        "income, debt, loans, and repayment history."
    )

    if model_name:
        st.subheader("🤖 Model in Use")
        st.write(f"**{model_name}** (selected automatically for having the "
                 f"best overall performance during training).")

    if metrics_df is not None:
        st.subheader("📊 Model Performance")
        st.dataframe(metrics_df.style.format("{:.2%}"))


# MAIN AREA - INPUT FORM
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
    annual_income = st.number_input(
        "Annual Income ($)", min_value=0, max_value=1_000_000, value=50000, step=1000
    )
    debt_amount = st.number_input(
        "Total Debt Amount ($)", min_value=0, max_value=1_000_000, value=5000, step=500
    )

with col2:
    loan_amount = st.number_input(
        "Loan Amount Requested ($)", min_value=0, max_value=1_000_000, value=10000, step=500
    )
    number_of_loans = st.number_input(
        "Number of Existing Loans", min_value=0, max_value=20, value=1, step=1
    )
    payment_history = st.selectbox(
        "Payment History", options=["Poor", "Average", "Good"], index=1
    )

st.write("")  # small spacer
predict_clicked = st.button("🔍 Predict Credit Risk", type="primary", use_container_width=True)


# PREDICTION
if predict_clicked:
    payment_history_encoded = payment_encoder.transform([payment_history])[0]

    input_data = pd.DataFrame([{
        "Age": age,
        "Annual_Income": annual_income,
        "Debt_Amount": debt_amount,
        "Loan_Amount": loan_amount,
        "Number_of_Loans": number_of_loans,
        "Payment_History": payment_history_encoded,
    }])[feature_columns]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = probabilities[prediction] * 100

    st.write("")
    if prediction == 0:
        st.markdown(f"""
        <div class="result-box-low">
            <h2>✅ Low Risk Customer</h2>
            <p>This customer is predicted to be a <b>Good</b> credit risk,
            with <b>{confidence:.1f}%</b> confidence.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-high">
            <h2>⚠️ High Risk Customer</h2>
            <p>This customer is predicted to be a <b>Bad</b> credit risk,
            with <b>{confidence:.1f}%</b> confidence.</p>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("See prediction probabilities"):
        st.write(f"Probability of Low Risk (Good): **{probabilities[0]*100:.1f}%**")
        st.write(f"Probability of High Risk (Bad): **{probabilities[1]*100:.1f}%**")
        st.progress(float(probabilities[1]))

    st.caption(
        "Disclaimer: this is a beginner educational project trained on "
        "synthetic data, not a real financial decision-making tool."
    )
