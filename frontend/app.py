import streamlit as st
import requests

# API URL
API_URL = "http://localhost:8000/predict" # For local testing
# In production, we would use the deployed Render API URL, e.g., "https://telco-churn-api.onrender.com/predict"

st.set_page_config(page_title="Telco Churn Predictor", page_icon="🔮", layout="wide")

# Custom CSS for a premium look
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Premium Headers */
    h1, h2, h3 {
        color: #2e3b4e;
        font-weight: 700 !important;
    }
    
    /* Stylish buttons */
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🔮 Telco Customer Churn Predictor")
st.markdown("[![View on GitHub](https://img.shields.io/badge/GitHub-View_Source_Code-181717?logo=github&style=for-the-badge)](https://github.com/arpitsingh6070/telco-churn-project)")
st.markdown("Predict whether a customer is likely to churn using our advanced machine learning model.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.subheader("Account Info")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=50.0)

with col3:
    st.subheader("Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.markdown("---")

if st.button("Predict Churn Risk"):
    # Build payload
    payload = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }
    
    with st.spinner("Analyzing customer profile..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                churn_pred = result.get("churn_prediction")
                churn_prob = result.get("churn_probability", 0.0)
                
                st.subheader("Prediction Results")
                r_col1, r_col2 = st.columns(2)
                
                if churn_pred == "Yes":
                    r_col1.error(f"⚠️ High Risk: Customer is likely to CHURN")
                else:
                    r_col1.success(f"✅ Safe: Customer is likely to STAY")
                    
                r_col2.metric("Churn Probability", f"{churn_prob:.1%}")
                
                st.progress(float(churn_prob))
                
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to the backend API. Is it running? Details: {e}")
