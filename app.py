import streamlit as st
import joblib
import numpy as np

model = joblib.load("model.joblib")

st.title("Student Performance Prediction")

hours_studied = st.number_input("Hours Studied", min_value=0, max_value=100)
previous_score = st.number_input("Previous Score", min_value=0, max_value=100)
sleep_hours = st.number_input("Hours of Sleep", min_value=0, max_value=100)
sample_paper = st.number_input("Sample Papers Solved", min_value=0, max_value=100)
coaching = st.radio("Has the student taken coaching?", ["Yes", "No"])

coaching_value = 1 if coaching == "Yes" else 0 

if st.button("Predict Performance"):
    
    input_data = np.array([[hours_studied, previous_score, sleep_hours, sample_paper, coaching_value]])

    try:
        prediction = model.predict(input_data)

        st.success(f"Predicted Performance: {prediction[0]:.2f}")
    
    except ValueError as e:
        st.error(f"Feature mismatch error: {e}")


