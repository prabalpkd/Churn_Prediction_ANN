import streamlit as st
import pandas as pd
import joblib
from tensorflow import keras

# ======================================================
# Page config (LANDSCAPE)
# ======================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)


# ======================================================
# Load artifacts
# ======================================================
@st.cache_resource
def load_artifacts():
    model = keras.models.load_model("churn_model.keras")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("model_features.pkl")
    mappings = joblib.load("mappings.pkl")
    return model, scaler, feature_names, mappings

model, scaler, feature_names, mappings = load_artifacts()

YES_NO_MAP = mappings["yes_no"]
GENDER_MAP = mappings["gender"]

# ======================================================
# Preprocessing
# ======================================================
def preprocess_input(df):

    df = df.replace('No internet service', 'No')
    df = df.replace('No phone service', 'No')

    yes_no_cols = [
        'Partner','Dependents','PhoneService','MultipleLines',
        'OnlineSecurity','OnlineBackup','DeviceProtection',
        'TechSupport','StreamingTV','StreamingMovies',
        'PaperlessBilling'
    ]

    for col in yes_no_cols:
        df[col] = df[col].map(YES_NO_MAP)

    df['gender'] = df['gender'].map(GENDER_MAP)

    df = pd.get_dummies(
        df,
        columns=['InternetService', 'Contract', 'PaymentMethod'],
        dtype=int
    )

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    df[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(
        df[['tenure','MonthlyCharges','TotalCharges']]
    )

    return df

# ======================================================
# Title
# ======================================================
st.markdown(
    "<h1 style='text-align: center;'>Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)

# ======================================================
# FORM
# ======================================================
with st.form("churn_form"):

    # ---------- ROW 1 ----------
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<div class="card"><b>👤 Customer</b>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", list(GENDER_MAP.keys()))
        Partner = st.selectbox("Partner", list(YES_NO_MAP.keys()))
        Dependents = st.selectbox("Dependents", list(YES_NO_MAP.keys()))
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><b>📞 Phone & Internet</b>', unsafe_allow_html=True)
        PhoneService = st.selectbox("Phone Service", list(YES_NO_MAP.keys()))
        MultipleLines = st.selectbox("Multiple Lines", list(YES_NO_MAP.keys()))
        InternetService = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card"><b>🧾 Contract</b>', unsafe_allow_html=True)
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        PaperlessBilling = st.selectbox("Paperless Billing", list(YES_NO_MAP.keys()))
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><b>💰 Charges</b>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (months)", 0, 72, 40)
        MonthlyCharges = st.slider("Monthly Charges", 0.0, 150.0, 70.0)
        TotalCharges = st.slider("Total Charges", 0.0, 8000.0, 1500.0)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- ROW 2 ----------
    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.markdown('<div class="card"><b>🔐 Security</b>', unsafe_allow_html=True)
        OnlineSecurity = st.selectbox("Online Security", list(YES_NO_MAP.keys()))
        OnlineBackup = st.selectbox("Online Backup", list(YES_NO_MAP.keys()))
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        st.markdown('<div class="card"><b>🛠 Support</b>', unsafe_allow_html=True)
        DeviceProtection = st.selectbox("Device Protection", list(YES_NO_MAP.keys()))
        TechSupport = st.selectbox("Tech Support", list(YES_NO_MAP.keys()))
        st.markdown('</div>', unsafe_allow_html=True)

    with c7:
        st.markdown('<div class="card"><b>📺 Streaming</b>', unsafe_allow_html=True)
        StreamingTV = st.selectbox("Streaming TV", list(YES_NO_MAP.keys()))
        StreamingMovies = st.selectbox("Streaming Movies", list(YES_NO_MAP.keys()))
        st.markdown('</div>', unsafe_allow_html=True)

    with c8:
        st.markdown('<div class="card"><b>▶ Action</b>', unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "🔮 Predict Churn Risk",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# Prediction (INLINE, NO SCROLL)
# ======================================================
if submitted:

    input_df = pd.DataFrame([{
        'gender': gender,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaymentMethod': PaymentMethod,
        'PaperlessBilling': PaperlessBilling,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    processed = preprocess_input(input_df)
    prob = model.predict(processed)[0][0]

    st.markdown("### 📊 Prediction Result")

    if prob > 0.45:
        st.error(f"⚠️ **High Risk of Churn**  \nProbability: **{prob:.2%}**")
    else:
        st.success(f"✅ **Low Risk of Churn**  \nProbability: **{prob:.2%}**")
