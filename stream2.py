# %%
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

# %%
# Load cancer data
def load_cancer_data():
    df = pd.read_csv('cancer.csv')
    return df

# Load diabetes data
def load_diabetes_data():
    df = pd.read_csv('diabetes.csv')
    return df

# Load the trained cancer diagnosis model
def load_cancer_model():
    model = tf.keras.models.load_model('cancer_model.h5')
    return model

# Load the trained diabetes prediction model
def load_diabetes_model():
    model = joblib.load('svm_diabetes_model.joblib')
    return model

# Make cancer diagnosis prediction
def predict_cancer_diagnosis(data, model):
    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Make predictions
    predictions = model.predict(scaled_data)

    return predictions

# Make diabetes prediction
def predict_diabetes(data, model):
    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Make predictions
    predictions = model.predict(scaled_data)

    return predictions

# Display the Cancer Diagnosis page
def display_cancer_diagnosis_page():
    st.title("Cancer Diagnosis Prediction")

    # Load cancer data
    data = load_cancer_data()

    # Load cancer model
    model = load_cancer_model()

    # Display input fields for cancer diagnosis
    st.markdown("### Input Fields")
    # Replace the following lines with the code to display input fields for cancer diagnosis
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
    # ...

    # Make cancer diagnosis prediction
    if st.button("Predict"):
        # Get user input
        input_data = np.array([[age, bmi]])  # Example input data

        # Make prediction
        prediction = predict_cancer_diagnosis(input_data, model)

        # Display prediction result
        st.subheader("Prediction Result:")
        st.write(prediction)

# Display the Diabetes Prediction page
def display_diabetes_prediction_page():
    st.title("Diabetes Prediction")

    # Load diabetes data
    data = load_diabetes_data()

    # Load diabetes model
    model = load_diabetes_model()

    # Display input fields for diabetes prediction
    st.markdown("### Input Fields")
    # Replace the following lines with the code to display input fields for diabetes prediction
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
    # ...

    # Make diabetes prediction
    if st.button("Predict"):
        # Get user input
        input_data = np.array([[age, bmi]])  # Example input data

        # Make prediction
        prediction = predict_diabetes(input_data, model)

        # Display prediction result
        st.subheader("Prediction Result:")
        st.write(prediction)

# Main function for the Streamlit app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ("Cancer Diagnosis", "Diabetes Prediction"))

    if page == "Cancer Diagnosis":
        display_cancer_diagnosis_page()
    elif page == "Diabetes Prediction":
        display_diabetes_prediction_page()

if __name__ == "__main__":
    main()


