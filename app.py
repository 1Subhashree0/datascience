import streamlit as st
import numpy as np
import pickle

# ===== Load model and scaler =====
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ===== App Title =====
st.title("Breast Cancer Survival Prediction (Logistic Model)")
st.write("This tool predicts the likelihood of a patient surviving breast cancer based on clinical factors.")

# ===== Input Fields =====
tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0, format="%.2f")
survival_months = st.number_input("Survival Months", min_value=0.0, format="%.1f")
age = st.number_input("Age", min_value=0, max_value=120)

estrogen_status = st.selectbox("Estrogen Status", ["Positive", "Negative"])
progesterone_status = st.selectbox("Progesterone Status", ["Positive", "Negative"])

# ===== Encode categorical inputs =====
estrogen = 1 if estrogen_status == "Positive" else 0
progesterone = 1 if progesterone_status == "Positive" else 0

# ===== Predict Button =====
if st.button("Predict"):
    # Format and scale user input
    user_input = np.array([[tumor_size, survival_months, age, estrogen, progesterone]])
    scaled_input = scaler.transform(user_input)

    # Predict probabilities and final class
    prob = model.predict_proba(scaled_input)[0]
    prediction = np.argmax(prob)  # This ensures the prediction matches highest probability

    # Show probabilities
    st.write(f" Probability of being DEAD: **{prob[1]:.2%}**")
    st.write(f" Probability of being ALIVE: **{prob[0]:.2%}**")

    # Show final result
    if prediction == 1:
        st.error(" Prediction: The patient is likely to be **DEAD**.")
    else:
        st.success(" Prediction: The patient is likely to be **ALIVE**.")
