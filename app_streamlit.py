import streamlit as st
import pandas as pd
import json
import os
import boto3
from botocore.exceptions import ClientError

ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(features: dict) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

ALL_FEATURES = [
    "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix",
    "Outstanding_Debt", "Credit_Utilization_Ratio",
    "Payment_of_Min_Amount", "Total_EMI_per_month",
    "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance",
    "Credit_History_Age_Months", "Num_Loan_Types",
]

OCCUPATION_OPTIONS = [
    "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur",
    "Journalist", "Lawyer", "Manager", "Mechanic", "Media_Manager",
    "Musician", "Scientist", "Teacher", "Writer",
]
CREDIT_MIX_OPTIONS = ["Bad", "Standard", "Good"]
PAYMENT_MIN_OPTIONS = ["Yes", "No", "NM"]
PAYMENT_BEHAVIOUR_OPTIONS = [
    "High_spent_Small_value_payments",
    "High_spent_Medium_value_payments",
    "High_spent_Large_value_payments",
    "Low_spent_Small_value_payments",
    "Low_spent_Medium_value_payments",
    "Low_spent_Large_value_payments",
]

TEST_CASES = {
    "🟢 Test Case: Good": {
        "Age": 45, "Occupation": "Doctor", "Annual_Income": 120000.0,
        "Monthly_Inhand_Salary": 9500.0, "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 3, "Interest_Rate": 5,
        "Num_of_Loan": 1, "Delay_from_due_date": 0,
        "Num_of_Delayed_Payment": 0, "Changed_Credit_Limit": 15.0,
        "Num_Credit_Inquiries": 1, "Credit_Mix": "Good",
        "Outstanding_Debt": 500.0, "Credit_Utilization_Ratio": 22.0,
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 50.0,
        "Amount_invested_monthly": 300.0,
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "Monthly_Balance": 600.0, "Credit_History_Age_Months": 250,
        "Num_Loan_Types": 1,
    },
    "🟡 Test Case: Standard": {
        "Age": 33, "Occupation": "Engineer", "Annual_Income": 45000.0,
        "Monthly_Inhand_Salary": 3500.0, "Num_Bank_Accounts": 5,
        "Num_Credit_Card": 5, "Interest_Rate": 14,
        "Num_of_Loan": 4, "Delay_from_due_date": 12,
        "Num_of_Delayed_Payment": 8, "Changed_Credit_Limit": 8.5,
        "Num_Credit_Inquiries": 5, "Credit_Mix": "Standard",
        "Outstanding_Debt": 2500.0, "Credit_Utilization_Ratio": 33.0,
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 150.0,
        "Amount_invested_monthly": 80.0,
        "Payment_Behaviour": "High_spent_Medium_value_payments",
        "Monthly_Balance": 250.0, "Credit_History_Age_Months": 120,
        "Num_Loan_Types": 4,
    },
    "🔴 Test Case: Poor": {
        "Age": 25, "Occupation": "Mechanic", "Annual_Income": 15000.0,
        "Monthly_Inhand_Salary": 1100.0, "Num_Bank_Accounts": 8,
        "Num_Credit_Card": 8, "Interest_Rate": 30,
        "Num_of_Loan": 8, "Delay_from_due_date": 40,
        "Num_of_Delayed_Payment": 20, "Changed_Credit_Limit": 3.0,
        "Num_Credit_Inquiries": 10, "Credit_Mix": "Bad",
        "Outstanding_Debt": 5000.0, "Credit_Utilization_Ratio": 45.0,
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 500.0,
        "Amount_invested_monthly": 20.0,
        "Payment_Behaviour": "High_spent_Large_value_payments",
        "Monthly_Balance": 50.0, "Credit_History_Age_Months": 24,
        "Num_Loan_Types": 8,
    },
}

CLASS_STYLES = {
    "Good": ("🟢", "#22c55e"),
    "Standard": ("🟡", "#eab308"),
    "Poor": ("🔴", "#ef4444"),
}

def main():
    st.set_page_config(
        page_title="Credit Score Prediction (AWS)",
        page_icon="☁️",
        layout="wide",
    )

    st.markdown("""
    <style>
        .main-header { text-align: center; padding: 1rem 0 0.5rem 0; }
        .prediction-box { text-align: center; padding: 2rem; border-radius: 12px; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>☁️ Credit Score Prediction (AWS Cloud)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Model di-deploy via AWS SageMaker Endpoint | Streamlit di EC2</p>", unsafe_allow_html=True)
    st.divider()

    st.sidebar.header("Quick Test Cases")
    st.sidebar.caption("Klik salah satu untuk mengisi form otomatis.")

    selected_test = None
    for test_name in TEST_CASES:
        if st.sidebar.button(test_name, use_container_width=True):
            selected_test = test_name

    if selected_test:
        st.session_state["test_values"] = TEST_CASES[selected_test]
        st.session_state["test_name"] = selected_test

    vals = st.session_state.get("test_values", {})

    st.subheader("Input Data Nasabah")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Informasi Pribadi**")
        age = st.number_input("Age", min_value=18, max_value=100,
                              value=int(vals.get("Age", 30)), step=1)
        occupation = st.selectbox("Occupation", OCCUPATION_OPTIONS,
                                  index=OCCUPATION_OPTIONS.index(vals["Occupation"]) if vals.get("Occupation") in OCCUPATION_OPTIONS else 0)
        annual_income = st.number_input("Annual Income ($)", min_value=0.0,
                                        value=float(vals.get("Annual_Income", 40000.0)), step=1000.0, format="%.2f")
        monthly_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0.0,
                                          value=float(vals.get("Monthly_Inhand_Salary", 3000.0)), step=100.0, format="%.2f")
        num_bank = st.number_input("Num Bank Accounts", min_value=0, max_value=20,
                                    value=int(vals.get("Num_Bank_Accounts", 3)), step=1)
        num_cc = st.number_input("Num Credit Card", min_value=0, max_value=20,
                                  value=int(vals.get("Num_Credit_Card", 3)), step=1)
        interest_rate = st.number_input("Interest Rate (%)", min_value=1, max_value=40,
                                         value=int(vals.get("Interest_Rate", 10)), step=1)

    with col2:
        st.markdown("**Informasi Pinjaman**")
        num_loan = st.number_input("Num of Loan", min_value=0, max_value=15,
                                    value=int(vals.get("Num_of_Loan", 2)), step=1)
        delay_due = st.number_input("Delay from Due Date (hari)", min_value=0, max_value=60,
                                     value=int(vals.get("Delay_from_due_date", 5)), step=1)
        num_delayed = st.number_input("Num of Delayed Payment", min_value=0, max_value=30,
                                       value=int(vals.get("Num_of_Delayed_Payment", 3)), step=1)
        changed_limit = st.number_input("Changed Credit Limit (%)", min_value=0.0,
                                         value=float(vals.get("Changed_Credit_Limit", 10.0)), step=0.5, format="%.2f")
        num_inquiry = st.number_input("Num Credit Inquiries", min_value=0, max_value=20,
                                       value=int(vals.get("Num_Credit_Inquiries", 3)), step=1)
        credit_mix = st.selectbox("Credit Mix", CREDIT_MIX_OPTIONS,
                                   index=CREDIT_MIX_OPTIONS.index(vals["Credit_Mix"]) if vals.get("Credit_Mix") in CREDIT_MIX_OPTIONS else 1)
        outstanding = st.number_input("Outstanding Debt ($)", min_value=0.0,
                                       value=float(vals.get("Outstanding_Debt", 1500.0)), step=100.0, format="%.2f")

    with col3:
        st.markdown("**Informasi Pembayaran**")
        util_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0,
                                      value=float(vals.get("Credit_Utilization_Ratio", 30.0)), step=1.0, format="%.2f")
        pay_min = st.selectbox("Payment of Min Amount", PAYMENT_MIN_OPTIONS,
                                index=PAYMENT_MIN_OPTIONS.index(vals["Payment_of_Min_Amount"]) if vals.get("Payment_of_Min_Amount") in PAYMENT_MIN_OPTIONS else 0)
        total_emi = st.number_input("Total EMI per Month ($)", min_value=0.0,
                                     value=float(vals.get("Total_EMI_per_month", 100.0)), step=10.0, format="%.2f")
        invest_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0.0,
                                          value=float(vals.get("Amount_invested_monthly", 100.0)), step=10.0, format="%.2f")
        pay_behaviour = st.selectbox("Payment Behaviour", PAYMENT_BEHAVIOUR_OPTIONS,
                                      index=PAYMENT_BEHAVIOUR_OPTIONS.index(vals["Payment_Behaviour"]) if vals.get("Payment_Behaviour") in PAYMENT_BEHAVIOUR_OPTIONS else 0)
        monthly_bal = st.number_input("Monthly Balance ($)", min_value=0.0,
                                       value=float(vals.get("Monthly_Balance", 300.0)), step=10.0, format="%.2f")
        credit_hist = st.number_input("Credit History Age (bulan)", min_value=0,
                                       value=int(vals.get("Credit_History_Age_Months", 100)), step=1)
        num_loan_types = st.number_input("Num Loan Types", min_value=0, max_value=15,
                                          value=int(vals.get("Num_Loan_Types", 2)), step=1)

    input_values = {
        "Age": age, "Occupation": occupation, "Annual_Income": annual_income,
        "Monthly_Inhand_Salary": monthly_salary, "Num_Bank_Accounts": num_bank,
        "Num_Credit_Card": num_cc, "Interest_Rate": interest_rate,
        "Num_of_Loan": num_loan, "Delay_from_due_date": delay_due,
        "Num_of_Delayed_Payment": num_delayed, "Changed_Credit_Limit": changed_limit,
        "Num_Credit_Inquiries": num_inquiry, "Credit_Mix": credit_mix,
        "Outstanding_Debt": outstanding, "Credit_Utilization_Ratio": util_ratio,
        "Payment_of_Min_Amount": pay_min, "Total_EMI_per_month": total_emi,
        "Amount_invested_monthly": invest_monthly, "Payment_Behaviour": pay_behaviour,
        "Monthly_Balance": monthly_bal, "Credit_History_Age_Months": credit_hist,
        "Num_Loan_Types": num_loan_types,
    }

    st.divider()

    if st.button("Prediksi Credit Score", type="primary", use_container_width=True):
        with st.spinner("Mengirim request ke SageMaker Endpoint..."):
            try:
                result = invoke_endpoint(input_values)
            except ClientError as e:
                st.error(f"Gagal menghubungi SageMaker Endpoint: {e}")
                return
            except Exception as e:
                st.error(f"Error: {e}")
                return

        prediction = result["predictions"][0]
        probabilities = result.get("probabilities", [[]])[0]
        classes = ["Good", "Poor", "Standard"]

        emoji, color = CLASS_STYLES.get(prediction, ("❓", "#6b7280"))

        st.markdown(f"""
        <div class='prediction-box' style='background: linear-gradient(135deg, {color}22, {color}11); border: 2px solid {color};'>
            <h2 style='margin:0; color:{color};'>{emoji} Hasil Prediksi</h2>
            <h1 style='margin:0.5rem 0; color:{color}; font-size:3rem;'>{prediction}</h1>
        </div>
        """, unsafe_allow_html=True)

        if probabilities:
            st.subheader("Probabilitas per Kelas")
            prob_dict = dict(zip(classes, probabilities))
            prob_cols = st.columns(3)
            for i, (cls, prob) in enumerate(sorted(prob_dict.items(), key=lambda x: -x[1])):
                cls_emoji, cls_color = CLASS_STYLES.get(cls, ("❓", "#6b7280"))
                with prob_cols[i]:
                    st.metric(label=f"{cls_emoji} {cls}", value=f"{prob * 100:.1f}%")
                    st.progress(prob)

        st.subheader("Data Input yang Digunakan")
        input_df = pd.DataFrame([input_values])
        st.dataframe(input_df, use_container_width=True)

        if "test_name" in st.session_state:
            st.info(f"Test case yang digunakan: **{st.session_state['test_name']}**")


if __name__ == "__main__":
    main()
