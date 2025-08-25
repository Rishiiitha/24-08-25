import streamlit as st
import pickle
import numpy as np

model, scaler = pickle.load(open("trained_model.sav", "rb"))

st.title("Credit Card Availability Prediction")
st.write("Enter user details below:")

Age = st.number_input("Age (years):", min_value=1, max_value=100, value=30)
Experience = st.number_input("Experience (years):", min_value=0, max_value=50, value=5)
Income = st.number_input("Income (LPA):", min_value=0, max_value=250, value=50)
Family = st.number_input("Family (count):", min_value=1, max_value=10, value=2)
CCAvg = st.number_input("Average Credit Card Spending:", min_value=0.0, max_value=10.0, value=1.0)
Education = st.selectbox("Education Level:", [1, 2, 3])  # 1=Undergrad, 2=Graduate, 3=Advanced/Professional
Mortgage = st.number_input("Mortgage:", min_value=0, max_value=650, value=0)
Personal_Loan = st.selectbox("Personal Loan:", [0, 1])  # 0=No, 1=Yes
Securities_Account = st.selectbox("Securities Account:", [0, 1])
CD_Account = st.selectbox("CD Account:", [0, 1])
Online = st.selectbox("Online Banking:", [0, 1])

features = np.array([[Age, Experience, Income, Family, CCAvg, Education,
                      Mortgage, Personal_Loan, Securities_Account, CD_Account, Online]])

features_scaled = scaler.transform(features)

if st.button("Predict Credit Card Availability"):
    prediction = model.predict(features_scaled)
    result = "Available ✅" if prediction[0] == 1 else "Not Available ❌"
    st.success(f"Credit card availability: {result}")

