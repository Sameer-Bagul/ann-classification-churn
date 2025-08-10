"""
Customer Churn Prediction Web Application

This Streamlit application predicts whether a bank customer is likely to churn (leave the bank)
using an Artificial Neural Network (ANN) trained on customer data.

Author: Sameer Bagul
Date: August 2025
Purpose: Educational demonstration of ML model deployment
"""

# ============================================================================
# IMPORT STATEMENTS
# ============================================================================

import streamlit as st              # Web application framework for ML apps
import numpy as np                  # Numerical computing library
import tensorflow as tf             # Deep learning framework
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd                 # Data manipulation and analysis
import pickle                       # For loading saved preprocessing objects
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL AND PREPROCESSING OBJECTS LOADING
# ============================================================================

@st.cache_resource  # Streamlit decorator to cache expensive operations
def load_model_and_preprocessors():
    """
    Load the trained model and preprocessing objects.
    
    This function uses Streamlit's caching to ensure that the model and 
    preprocessors are loaded only once, improving app performance.
    
    Returns:
        tuple: (model, label_encoder_gender, onehot_encoder_geo, scaler)
    """
    
    # Load the trained Neural Network model
    # The .h5 format is HDF5, commonly used for saving Keras/TensorFlow models
    model = tf.keras.models.load_model('model.h5')
    
    # Load the Label Encoder for Gender
    # LabelEncoder converts categorical text to numerical values
    # Example: 'Male' -> 1, 'Female' -> 0
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    # Load the One-Hot Encoder for Geography
    # OneHotEncoder creates binary columns for each category
    # Example: 'France' -> [1, 0, 0], 'Germany' -> [0, 1, 0], 'Spain' -> [0, 0, 1]
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    # Load the Standard Scaler
    # StandardScaler normalizes features to have mean=0 and standard deviation=1
    # This ensures all features are on the same scale for the neural network
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return model, label_encoder_gender, onehot_encoder_geo, scaler

# Load all components
model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_preprocessors()

# ============================================================================
# STREAMLIT WEB APPLICATION INTERFACE
# ============================================================================

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title with custom styling
st.title('🏦 Customer Churn Prediction System')
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Information section
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    This application uses an **Artificial Neural Network (ANN)** to predict whether a bank customer 
    is likely to churn (stop using the bank's services). 
    
    **How it works:**
    1. Enter customer information in the form below
    2. The model processes the data using the same preprocessing steps used during training
    3. A trained neural network predicts the churn probability
    4. The result is displayed with an interpretation
    
    **Model Details:**
    - Architecture: 2 hidden layers (64 and 32 neurons)
    - Activation: ReLU for hidden layers, Sigmoid for output
    - Training Accuracy: ~86%
    - Features: 12 customer characteristics
    """)

# ============================================================================
# USER INPUT SECTION
# ============================================================================

st.header("📝 Customer Information")
st.markdown("Please enter the customer details below:")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Demographics")
    
    # Geography Selection
    # We extract available categories from the fitted encoder
    geography = st.selectbox(
        'Country/Geography', 
        onehot_encoder_geo.categories_[0],  # Get the categories from fitted encoder
        help="The country where the customer is located. Different regions may have different churn patterns."
    )
    
    # Gender Selection
    gender = st.selectbox(
        'Gender', 
        label_encoder_gender.classes_,  # Get classes from fitted encoder
        help="Customer's gender. This may correlate with different banking preferences."
    )
    
    # Age Input
    age = st.slider(
        'Age', 
        min_value=18, 
        max_value=92, 
        value=35,
        help="Customer's age in years. Age groups may have different churn behaviors."
    )
    
    # Tenure (Years as Customer)
    tenure = st.slider(
        'Tenure (Years with Bank)', 
        min_value=0, 
        max_value=10, 
        value=5,
        help="Number of years the customer has been with the bank. Longer tenure usually means lower churn."
    )

with col2:
    st.subheader("💰 Financial Information")
    
    # Credit Score Input
    credit_score = st.number_input(
        'Credit Score', 
        min_value=300, 
        max_value=850, 
        value=650,
        help="Customer's credit score (300-850). Higher scores indicate better creditworthiness."
    )
    
    # Account Balance
    balance = st.number_input(
        'Account Balance ($)', 
        min_value=0.0, 
        value=50000.0,
        step=1000.0,
        help="Current account balance. Customers with higher balances are typically more valuable."
    )
    
    # Estimated Salary
    estimated_salary = st.number_input(
        'Estimated Annual Salary ($)', 
        min_value=0.0, 
        value=75000.0,
        step=5000.0,
        help="Customer's estimated annual income. Income level affects banking needs and churn risk."
    )

# Banking Products and Services
st.subheader("🏦 Banking Relationship")

col3, col4 = st.columns(2)

with col3:
    # Number of Products
    num_of_products = st.slider(
        'Number of Bank Products Used', 
        min_value=1, 
        max_value=4, 
        value=2,
        help="Total number of bank products (accounts, cards, loans, etc.). More products create switching barriers."
    )
    
    # Credit Card Ownership
    has_cr_card = st.selectbox(
        'Has Credit Card', 
        options=[0, 1],
        format_func=lambda x: 'Yes' if x == 1 else 'No',
        help="Whether the customer has a credit card with the bank."
    )

with col4:
    # Active Member Status
    is_active_member = st.selectbox(
        'Is Active Member', 
        options=[0, 1],
        format_func=lambda x: 'Yes' if x == 1 else 'No',
        help="Whether the customer actively uses bank services. Active customers are less likely to churn."
    )

# ============================================================================
# DATA PREPROCESSING AND PREDICTION
# ============================================================================

# Add a predict button for better user experience
if st.button("🔮 Predict Churn Probability", type="primary"):
    
    # Create a progress bar for user feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Prepare the input data (25% progress)
    status_text.text('Preparing input data...')
    progress_bar.progress(25)
    
    # Create a DataFrame with the input data
    # This matches the structure used during model training
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],  # Convert text to number
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })
    
    # Step 2: One-hot encode Geography (50% progress)
    status_text.text('Encoding categorical variables...')
    progress_bar.progress(50)
    
    # Transform geography using the fitted encoder
    # This creates binary columns for each country
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, 
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )
    
    # Combine the encoded geography with other features
    # reset_index(drop=True) ensures proper concatenation
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    
    # Step 3: Scale the features (75% progress)
    status_text.text('Scaling features...')
    progress_bar.progress(75)

    # Apply the same scaling used during training
    # This ensures features have the same distribution as training data
    # Fix: Ensure input_data columns match scaler's expected features
    # Drop target column if present
    target_col = 'EstimatedSalary'
    if target_col in input_data.columns and target_col not in scaler.feature_names_in_:
        input_data = input_data.drop(target_col, axis=1)

    # Add missing columns with default value (0)
    for col in scaler.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match scaler
    input_data = input_data[scaler.feature_names_in_]

    input_data_scaled = scaler.transform(input_data)
    
    # Step 4: Make prediction (100% progress)
    status_text.text('Generating prediction...')
    progress_bar.progress(100)
    
    # Use the trained model to predict churn probability
    # model.predict() returns a 2D array, so we extract the probability
    prediction = model.predict(input_data_scaled, verbose=0)  # verbose=0 suppresses output
    prediction_proba = prediction[0][0]  # Extract probability from array
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # ============================================================================
    # RESULTS DISPLAY
    # ============================================================================
    
    st.header("🎯 Prediction Results")
    
    # Create columns for result display
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        # Display probability as a large metric
        st.metric(
            label="Churn Probability", 
            value=f"{prediction_proba:.1%}",
            help="Probability that this customer will churn (leave the bank)"
        )
    
    with result_col2:
        # Determine risk level and color
        if prediction_proba < 0.3:
            risk_level = "Low Risk"
            color = "green"
            recommendation = "Customer is likely to stay. Continue standard service."
        elif prediction_proba < 0.7:
            risk_level = "Medium Risk"
            color = "orange"
            recommendation = "Monitor customer satisfaction. Consider proactive engagement."
        else:
            risk_level = "High Risk"
            color = "red"
            recommendation = "Immediate retention efforts recommended. Contact customer service."
        
        # Display risk assessment
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 0.5rem;">
            <h3>Risk Level: {risk_level}</h3>
            <p><strong>Recommendation:</strong> {recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed interpretation
    st.subheader("📊 Detailed Analysis")
    
    # Business interpretation based on probability ranges
    if prediction_proba > 0.5:
        st.error(f"""
        **High Churn Risk (Probability: {prediction_proba:.1%})**
        
        This customer shows significant risk of churning. Consider:
        - Personal outreach from customer service
        - Special offers or incentives
        - Review of current banking products and services
        - Customer satisfaction survey
        """)
    else:
        st.success(f"""
        **Low Churn Risk (Probability: {prediction_proba:.1%})**
        
        This customer is likely to remain with the bank. Actions:
        - Continue current service level
        - Consider upselling additional products
        - Maintain regular communication
        - Use as reference for satisfaction surveys
        """)
    
    # Technical details for learning purposes
    with st.expander("🔍 Technical Details (For Learning)"):
        st.markdown(f"""
        **Model Processing Steps:**
        1. **Input Data Shape:** {input_data.shape}
        2. **Features After Encoding:** {list(input_data.columns)}
        3. **Scaled Data Range:** [{input_data_scaled.min():.3f}, {input_data_scaled.max():.3f}]
        4. **Model Architecture:** Input(12) → Dense(64) → Dense(32) → Dense(1)
        5. **Activation Functions:** ReLU → ReLU → Sigmoid
        6. **Raw Prediction:** {prediction[0][0]:.6f}
        
        **Feature Importance Notes:**
        - Age and tenure are typically strong predictors
        - Number of products creates switching barriers
        - Active membership status is highly influential
        - Geographic location may indicate market conditions
        """)
        
        # Show the processed data
        st.markdown("**Processed Input Data:**")
        st.dataframe(input_data, use_container_width=True)

# ============================================================================
# EDUCATIONAL SECTION
# ============================================================================

st.header("📚 Understanding the Model")

# Create tabs for different educational content
tab1, tab2, tab3 = st.tabs(["🧠 Neural Network", "📊 Data Processing", "💼 Business Impact"])

with tab1:
    st.markdown("""
    **How the Artificial Neural Network Works:**
    
    1. **Input Layer (12 neurons):** Receives all customer features
    2. **Hidden Layer 1 (64 neurons):** Learns complex patterns using ReLU activation
    3. **Hidden Layer 2 (32 neurons):** Refines patterns and reduces complexity
    4. **Output Layer (1 neuron):** Produces churn probability using Sigmoid activation
    
    **Activation Functions:**
    - **ReLU (Rectified Linear Unit):** f(x) = max(0, x) - Helps with gradient flow
    - **Sigmoid:** f(x) = 1/(1 + e^(-x)) - Outputs probability between 0 and 1
    
    **Training Process:**
    - **Forward Pass:** Data flows through network to produce prediction
    - **Loss Calculation:** Binary crossentropy measures prediction error
    - **Backpropagation:** Gradients flow backward to update weights
    - **Optimization:** Adam optimizer adjusts learning rate automatically
    """)

with tab2:
    st.markdown("""
    **Data Preprocessing Pipeline:**
    
    1. **Categorical Encoding:**
       - Gender: Male=1, Female=0 (Label Encoding)
       - Geography: Creates 3 binary columns (One-Hot Encoding)
    
    2. **Feature Scaling:**
       - StandardScaler: (value - mean) / standard_deviation
       - Ensures all features have mean=0, std=1
       - Prevents larger values from dominating smaller ones
    
    3. **Feature Engineering:**
       - Numerical features used directly
       - Categorical features transformed to numerical
       - All features scaled to same range
    
    **Why Preprocessing Matters:**
    - Neural networks are sensitive to input scale
    - Categorical data must be numerical
    - Consistent preprocessing ensures reliable predictions
    """)

with tab3:
    st.markdown("""
    **Business Value and Applications:**
    
    **Cost-Benefit Analysis:**
    - Customer acquisition cost: 5-25x more expensive than retention
    - Average customer lifetime value: $1,000-$10,000
    - Retention program cost: $50-$200 per customer
    - ROI of churn prevention: 300-500%
    
    **Actionable Insights:**
    - **High-risk customers:** Immediate intervention required
    - **Medium-risk customers:** Proactive engagement strategies  
    - **Low-risk customers:** Upselling opportunities
    
    **Implementation Strategy:**
    - Batch predictions for entire customer base
    - Real-time scoring for new customers
    - Integration with CRM systems
    - A/B testing of retention strategies
    
    **Success Metrics:**
    - Reduction in churn rate
    - Increase in customer lifetime value
    - Improvement in retention campaign effectiveness
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚀 Built with Streamlit and TensorFlow | 
    📚 Educational Project for Machine Learning | 
    🤖 Artificial Neural Network Implementation</p>
    <p><em>This application demonstrates the complete ML pipeline from data preprocessing to model deployment.</em></p>
</div>
""", unsafe_allow_html=True)
