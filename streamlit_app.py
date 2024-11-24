import streamlit as st
import pickle
import numpy as np

# Load your pre-trained model
with open("rf_model.pkl", 'rb') as file:
    model = pickle.load(file)

# Define the app layout
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSByQqTdmeNAG0Fhb7TAVN2X8BM9BOX6g0A0g&s")

st.title("Heart Disease Prediction App")

st.write("""
This app predicts the likelihood of a heart disease based on various input features.
""")

# Collect user inputs for the features you used in your model
age = st.number_input("Age", min_value=0, max_value=120)

gender = st.selectbox("Gender", ["0", "1"])
st.write("Male= 1, Female=0")

cp = st.selectbox("Chest Pain type", ["0", "1","2","	3"])

trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=500)

chol= st.number_input("Serum Cholesterol in mg/dL ", min_value=0, max_value=500)

fbs= st.selectbox("Fasting blood glucose > 120 mg/dL", ["1", "0"])

restecg= st.selectbox("resting electrocardiographic results", ["0", "1","2"])

thalach= st.number_input("Maximum heart rate by stress test", min_value=0, max_value=500)

exang= st.selectbox("Exercise induced angina", ["0", "1"])

oldpeak= st.number_input("ST depression", min_value=0, max_value=500)

slope = st.selectbox("the slope of the peak exercise ST segment", ["0", "1","2"])

ca= st.number_input("number of major vessels", min_value=0, max_value=3)

thal = st.selectbox("Thalius test result", ["0", "1","2","3"])

# Prepare input data for the model
input_data = np.array([[
    age, gender,cp, trestbps ,chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
]])

# Make prediction
if st.button("Predict Heart Disease Risk"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]  # Assuming model has predict_proba

    if prediction == 1:
        st.write("### Result: High risk of Myocardial Infarction")
    else:
        st.write("### Result: Low risk of Myocardial Infarction")
    st.write(f"**Probability of Myocardial Infarction**: {prediction_proba * 100:.2f}%")

