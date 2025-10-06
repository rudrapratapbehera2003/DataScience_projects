import streamlit as st
import numpy as np
import pickle


with open("log_reg_diabetes.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Diabetes Prediction App")

st.write("Enter patient details to predict the probability of diabetes")


preg = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, step=1)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)
skin = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, step=1)


if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.write(f"Predicted Probability of Diabetes: **{prob:.2f}**")
    if pred == 1:
        st.error("High risk of Diabetes")
    else:
        st.success("Low risk of Diabetes")
