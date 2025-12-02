# app.py
import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_diabetes

# Load the dataset to get feature names for input labels
diabetes_data = load_diabetes()
feature_names = diabetes_data.feature_names

# 1. Load the saved classification model and scaler
try:
    classifier_model = joblib.load('diabetes_classifier_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Error: Model or scaler file not found. Please ensure 'diabetes_classifier_model.joblib' and 'scaler.joblib' are in the same directory.")
    st.stop()

# 2. Streamlit Application Title and Description
st.title('Diabetes Prediction App')
st.write('Enter the patient\u2019s details to predict whether they are diabetic or not.')

# Input fields for each feature
user_inputs = {}
for feature in feature_names:
    user_inputs[feature] = st.number_input(f'Enter {feature.replace("-", " ").title()}', value=0.0, step=0.01)

# Predict button
if st.button('Predict'):
    # 3. Collect user inputs into a NumPy array
    input_features = np.array([user_inputs[feature] for feature in feature_names]).reshape(1, -1)

    # 4. Scale the user input
    scaled_input = scaler.transform(input_features)

    # 5. Make a prediction
    prediction = classifier_model.predict(scaled_input)

    # 6. Display the prediction result
    if prediction[0] == 1:
        st.write('### Prediction: Diabetic')
        st.warning('Based on the provided data, the model predicts the person is likely Diabetic.')
    else:
        st.write('### Prediction: Not Diabetic')
        st.success('Based on the provided data, the model predicts the person is likely Not Diabetic.')

st.markdown("""
### Feature Descriptions:
- **Age**: Age in years
- **Sex**: Biological sex
- **Bmi**: Body mass index
- **Bp**: Average blood pressure
- **S1, S2, S3, S4, S5, S6**: Six blood serum measurements
""")
