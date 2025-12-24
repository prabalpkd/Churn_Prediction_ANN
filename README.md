# 📉 Customer Churn Prediction using Deep Learning

A complete **end-to-end Machine Learning project** that predicts whether a telecom customer is likely to **churn (leave the service)** using customer demographics, account information, and service usage data.

The project includes:
- Data preprocessing & feature engineering
- Artificial Neural Network (ANN) model
- Model evaluation with business-focused metrics
- Interactive **Streamlit web application**
- Ready-to-deploy pipeline

---

## 🚀 Live Application
> https://churnpredictionann-qyy8ygkdw3sknyhfw9fm6n.streamlit.app/

---

## 📊 Problem Statement

Customer churn is one of the most critical challenges faced by subscription-based businesses such as telecom companies.  
Acquiring a new customer is **5–7× more expensive** than retaining an existing one.

**Goal:**  
Predict whether a customer will churn (`Yes` / `No`) so that proactive retention strategies (discounts, offers, support) can be applied **before the customer leaves**.

---

## 🧠 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Target Variable:** `Churn` (Yes / No)
- **Total Records:** ~7,000 customers
- **Features Include:**
  - Demographics (gender, dependents, partner)
  - Services (internet, phone, streaming, security)
  - Account info (tenure, contract, payment method)
  - Charges (monthly & total)

---

## ⚙️ Data Preprocessing

Key preprocessing steps:
- Removed `customerID`
- Converted `TotalCharges` from string to numeric
- Handled missing values using **median imputation**
- Converted categorical variables (`Yes/No`) to binary
- One-hot encoded multi-class features:
  - InternetService
  - Contract
  - PaymentMethod
- Feature scaling using **MinMaxScaler**

All preprocessing artifacts are **saved and reused** in the Streamlit app to ensure consistency.

---

## 🤖 Model Architecture

An **Artificial Neural Network (ANN)** built using TensorFlow / Keras:

- Input Layer: 26 features
- Hidden Layers:
  - Dense (32 neurons, ReLU)
  - Dropout (0.2)
  - Dense (16 neurons, ReLU)
  - Dropout (0.2)
- Output Layer:
  - Dense (1 neuron, Sigmoid)

**Loss Function:** Binary Cross-Entropy  
**Optimizer:** Adam  
**Class Imbalance Handling:** Class Weights

---

## 📈 Model Evaluation

The trained model was evaluated on an unseen test dataset using multiple classification metrics to ensure both statistical validity and business relevance.

### 🔢 Performance Metrics

- **Accuracy:** **75%**
- **Recall (Churn = Yes):** **83%**
- **Precision (Churn = Yes):** Moderate, reflecting a recall-focused optimization
- **F1-score:** Balanced performance across classes

A custom decision threshold (`0.45`) was applied instead of the default `0.5` to improve the detection of churn-prone customers.

---

## 🔍 Why Recall Is More Important Than Accuracy

While the model achieves an overall **accuracy of 75%**, accuracy alone is not sufficient for churn prediction tasks due to the **asymmetric cost of misclassification**.

### 🎯 Business Rationale

| Prediction Outcome | Business Impact |
|--------------------|-----------------|
| False Positive (predict churn, customer stays) | Low cost (unnecessary retention offer) |
| **False Negative (predict no churn, customer actually churns)** | **High cost (customer and revenue loss)** |

### 📌 Emphasis on Recall

- A **recall of 83% for the churn class** indicates that the model successfully identifies **83 out of every 100 customers who actually churn**
- Missing a churner results in **no retention action**, leading to irreversible customer loss
- From a business standpoint, it is preferable to flag more potential churners, even at the cost of some false positives

➡️ For these reasons, this project **prioritizes recall over raw accuracy**, ensuring better alignment with real-world customer retention strategies.

---

## 🧠 Optimization Strategy

- Class imbalance addressed using **class weights**
- Decision threshold tuned to maximize recall
- Dropout layers applied to reduce overfitting
- Evaluation focused on **positive (churn) class performance**

---

## 🖥️ Streamlit Web Application

Features:
- Clean, landscape-oriented UI
- Dropdowns & sliders for customer attributes
- Real-time churn probability prediction
- Consistent preprocessing with saved artifacts

Saved files used in the app:
- `churn_model.keras` → trained ANN model
- `scaler.pkl` → feature scaler
- `model_features.pkl` → feature order
- `mappings.pkl` → categorical mappings

---

## 📁 Project Structure

Churn_Prediction/
│
├── app.py # Streamlit application
├── churn_model.keras # Trained ANN model
├── scaler.pkl # MinMaxScaler
├── model_features.pkl # Feature ordering
├── mappings.pkl # Encoding mappings
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── Churn_Prediction.ipynb # Model training notebook


---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Git & GitHub

---

## 📌 Future Improvements

- SHAP-based model explainability
- Cost-sensitive learning
- Threshold tuning based on business cost
- API deployment (FastAPI)
- Integration with CRM systems

---

## 👤 Author

**Prabal Kumar Deka**  
Machine Learning & Data Science Enthusiast  

---

## ⭐ Acknowledgements

- IBM Analytics for Telco Churn Dataset
- Streamlit for rapid ML deployment
- TensorFlow & Scikit-learn communities

