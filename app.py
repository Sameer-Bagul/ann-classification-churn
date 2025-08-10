import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# Load the trained model and preprocessors
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('feature_order.pkl', 'rb') as f:
    feature_order = pickle.load(f)  # List of feature names in correct order

# Streamlit UI
st.title("Customer Churn Prediction")

# Inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92, 30)
balance = st.number_input('Balance', min_value=0.0, step=0.01, format="%.2f")
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, step=1)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, step=0.01, format="%.2f")
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input DataFrame
input_dict = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
}

input_df = pd.DataFrame([input_dict])

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine
input_df = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)

# Add missing columns with zeros if needed
for col in feature_order:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns exactly as in training
input_df = input_df[feature_order]

# Scale
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
churn_prob = prediction[0][0]

st.write(f"Churn Probability: {churn_prob:.2f}")

if churn_prob > 0.5:
    st.error("The customer is likely to churn.")
else:
    st.success("The customer is not likely to churn.")
