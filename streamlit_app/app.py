import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, predictpipeline

# Streamlit App Title
st.title("Student Exam Performance Prediction")

st.write("Enter the student details to predict their exam score:")

# Input fields
gender = st.selectbox("Gender", ["male", "female"])
race_ethnicity = st.selectbox("Race/Ethnicity", [
                              "group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)

# Predict button
if st.button("Predict"):
    # Create data object
    data = CustomData(
        gender=gender,
        **{"race/ethnicity": race_ethnicity},
        **{"parental level of education": parental_education},
        lunch=lunch,
        **{"test preparation course": test_preparation},
        **{"reading score": reading_score},
        **{"writing score": writing_score}
   )

    df = data.get_data_as_data_frame()

    pipeline = predictpipeline()
    result = pipeline.predict(df)

    st.success(f"🎯 Predicted Math Score: {round(result[0], 2)}")
