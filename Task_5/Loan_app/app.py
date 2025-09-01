import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved pipeline (trained in your notebook)
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💰 Loan Default Prediction App")
st.write("Answer the questions below to check if a loan is **likely to default or not**.")

# --- Collect user inputs ---
sub_grade = st.text_input("Sub Grade (e.g., A1, B2, C3)", value="A1")
fico_score = st.number_input("FICO Score", min_value=300, max_value=850, value=700)
annual_inc = st.number_input("Annual Income ($)", min_value=0, step=1000, value=50000)
mort_acc = st.number_input("Mortgage Accounts", min_value=0, step=1, value=1)
initial_list_status = st.selectbox("Initial List Status", ["w", "f"])
time_to_earliest_cr_line = st.number_input("Time to Earliest Credit Line (months)", min_value=0, value=100)
emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
revol_bal = st.number_input("Revolving Balance ($)", min_value=0, step=500, value=10000)
term = st.selectbox("Loan Term", ["36 months", "60 months"])
home_ownership = st.selectbox("Home Ownership", ["OWN", "MORTGAGE", "RENT", "OTHER"])
revol_util = st.number_input("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=30.0)
dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=15.0)
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=40.0, value=10.0)
loan_amnt = st.number_input("Loan Amount ($)", min_value=0, step=500, value=15000)
verification_status = st.selectbox("Verification Status", ["Verified", "Source Verified", "Not Verified"])

# --- Feature engineering (same as notebook) ---
debt_to_income = loan_amnt / (annual_inc + 1e-6)
available_revol_credit = revol_bal / (revol_util + 1e-6)

# Put inputs into a dataframe (must match training features!)
input_data = pd.DataFrame([{
    "sub_grade": sub_grade,
    "fico_score": fico_score,
    "annual_inc": annual_inc,
    "mort_acc": mort_acc,
    "initial_list_status": initial_list_status,
    "time_to_earliest_cr_line": time_to_earliest_cr_line,
    "emp_length": emp_length,
    "revol_bal": revol_bal,
    "term": term,
    "home_ownership": home_ownership,
    "revol_util": revol_util,
    "dti": dti,
    "int_rate": int_rate,
    "loan_amnt": loan_amnt,
    "verification_status": verification_status,
    "debt_to_income": debt_to_income,
    "available_revol_credit": available_revol_credit
}])

# --- Predict ---
if st.button("Predict"):
    prediction = model.predict(input_data)[0]   # 0 = No Default, 1 = Default
    proba = model.predict_proba(input_data)[0][1]

    if proba > 0.9:
        st.error(f"⚠️ Loan is **Approved** (probability: {proba:.2f})")
    else:
        st.success(f"✅ Loan is **Not Approved** (probability of default: {proba:.2f})")
