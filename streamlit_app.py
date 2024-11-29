import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load your pre-trained model
with open("rf_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Define the app layout
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSByQqTdmeNAG0Fhb7TAVN2X8BM9BOX6g0A0g&s")

st.title("Hep-C Cirrhosis Prediction App")

st.write("""
This app can be used to detect the presence of cirrhosis in hepatitis C patients.
""")

# Collect user inputs for the features you used in your model
age = st.number_input("Age", min_value=0, max_value=120)

gender = st.selectbox("Gender", ["0", "1"])
st.write("Male= 0, Female=1")

BMI = st.number_input("Body mass index (BMI)", min_value=0, max_value=500)

Fever = st.selectbox("Fever", ["0", "1"])

NV = st.selectbox("Nausea/Vomiting", ["0", "1"])
Head = st.selectbox("Headache", ["0", "1"])
Dia = st.selectbox("Diarrhea", ["0", "1"])
Fatigue = st.selectbox("Fatigue", ["0", "1"])
J = st.selectbox("Jaundice", ["0", "1"])
EGP  =  st.selectbox("Epigastric pain", ["0", "1"])

WBC = st.number_input("WBC count")
RBC = st.number_input("RBC count")
HGB = st.number_input("Hemoglobin (HGB)")
Plat = st.number_input("Platelet count")
AST = st.number_input("AST")
ALT = st.number_input("ALT")
RNA = st.number_input("RNA count")
BHG = st.number_input("Baseline Histological Grading")

# Prepare input data for the model

s = MinMaxScaler()
input_data = np.array([[
    age, gender,BMI, Fever ,NV, Head,Dia, Fatigue,J, EGP, WBC, RBC, HGB, Plat, AST, ALT, RNA, BHG
]])

input_data = s.fit_transform(input_data)

# Make prediction
if st.button("Predict Cirrhosis risk"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]  # Assuming model has predict_proba

    if prediction == 1:
        st.write("### Result: High risk of developing cirrhosis")
    else:
        st.write("### Result: Low risk of developing cirrhosis")
    st.write(f"**Probability of Cirrhosis**: {prediction_proba * 100:.2f}%")

