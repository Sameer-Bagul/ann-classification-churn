# Customer Churn Prediction using Artificial Neural Networks (ANN)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Understanding the Problem](#understanding-the-problem)
- [Dataset Description](#dataset-description)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Artificial Neural Networks Explained](#artificial-neural-networks-explained)
- [Project Structure](#project-structure)
- [Installation and Setup](#installation-and-setup)
- [How to Run the Project](#how-to-run-the-project)
- [Technical Implementation](#technical-implementation)
- [Model Performance](#model-performance)
- [Web Application Features](#web-application-features)
- [Learning Outcomes](#learning-outcomes)
- [Advanced Topics](#advanced-topics)

## 🎯 Project Overview

This project demonstrates how to build and deploy a **Customer Churn Prediction** model using **Artificial Neural Networks (ANN)**. Customer churn prediction is a critical business problem where companies try to identify customers who are likely to stop using their services.

### What You'll Learn:
- Data preprocessing and feature engineering
- Artificial Neural Network architecture and implementation
- Model training with TensorFlow/Keras
- Hyperparameter tuning techniques
- Model deployment with Streamlit
- Real-world machine learning pipeline

## 🔍 Understanding the Problem

### What is Customer Churn?
**Customer Churn** (also called customer attrition) refers to when customers stop doing business with a company. In this context, it means bank customers closing their accounts or becoming inactive.

### Why is Churn Prediction Important?
1. **Cost Efficiency**: It's 5-25x more expensive to acquire new customers than retain existing ones
2. **Revenue Protection**: Preventing churn directly protects revenue streams
3. **Strategic Planning**: Helps in resource allocation and marketing strategies
4. **Customer Satisfaction**: Identifies pain points in customer experience

### Business Impact:
- **Proactive Retention**: Identify at-risk customers before they churn
- **Targeted Marketing**: Focus retention efforts on high-risk customers
- **Resource Optimization**: Allocate customer service resources efficiently

## 📊 Dataset Description

### Dataset: Bank Customer Churn Modelling
- **Source**: Simulated bank customer data
- **Size**: 10,000 customer records
- **Type**: Supervised learning (Classification problem)

### Features Explained:

| Feature | Type | Description | Business Relevance |
|---------|------|-------------|-------------------|
| `RowNumber` | Integer | Sequential ID | Not useful for prediction |
| `CustomerId` | Integer | Unique customer identifier | Not useful for prediction |
| `Surname` | String | Customer's last name | Not useful for prediction |
| `CreditScore` | Integer (350-850) | Credit score rating | Higher scores indicate better creditworthiness |
| `Geography` | Categorical | Country (France, Spain, Germany) | Different markets may have different churn patterns |
| `Gender` | Categorical | Male/Female | Gender-based preferences might affect churn |
| `Age` | Integer (18-92) | Customer's age | Different age groups have different banking needs |
| `Tenure` | Integer (0-10) | Years as bank customer | Longer tenure usually means lower churn probability |
| `Balance` | Float | Account balance | Customers with higher balances are valuable to retain |
| `NumOfProducts` | Integer (1-4) | Number of bank products used | More products create switching barriers |
| `HasCrCard` | Binary (0/1) | Has credit card | Additional product relationship |
| `IsActiveMember` | Binary (0/1) | Actively uses bank services | Active customers are less likely to churn |
| `EstimatedSalary` | Float | Annual salary estimate | Income level affects banking needs |
| `Exited` | Binary (0/1) | **TARGET**: Did customer churn? | What we're trying to predict |

## 🔄 Machine Learning Pipeline

### 1. Data Preprocessing
```
Raw Data → Cleaning → Feature Engineering → Encoding → Scaling → Model Ready Data
```

**Steps Explained:**
- **Data Cleaning**: Remove irrelevant columns (RowNumber, CustomerId, Surname)
- **Categorical Encoding**: Convert text categories to numbers
  - **Label Encoding**: For binary categories (Gender: Male=1, Female=0)
  - **One-Hot Encoding**: For multi-category features (Geography: creates separate columns)
- **Feature Scaling**: Normalize numerical values using StandardScaler
- **Train-Test Split**: Divide data for training (80%) and testing (20%)

### 2. Model Development
```
Architecture Design → Training → Validation → Hyperparameter Tuning → Final Model
```

### 3. Model Deployment
```
Trained Model → Web Interface → User Input → Prediction → Business Action
```

## 🧠 Artificial Neural Networks Explained

### What is an Artificial Neural Network?
An ANN is a computing system inspired by biological neural networks. It consists of interconnected nodes (neurons) that process information.

### Key Components:

#### 1. **Neurons (Nodes)**
- Basic processing units that receive inputs, apply weights, and produce outputs
- Each neuron applies an activation function to determine its output

#### 2. **Layers**
- **Input Layer**: Receives the initial data (our 12 features)
- **Hidden Layers**: Process the data through weighted connections
- **Output Layer**: Produces the final prediction (churn probability)

#### 3. **Weights and Biases**
- **Weights**: Determine the strength of connections between neurons
- **Biases**: Allow shifting the activation function for better fitting

#### 4. **Activation Functions**
- **ReLU (Rectified Linear Unit)**: `f(x) = max(0, x)` - Used in hidden layers
- **Sigmoid**: `f(x) = 1/(1 + e^(-x))` - Used in output layer for probability

### Our Model Architecture:
```
Input Layer (12 features) 
    ↓
Hidden Layer 1 (64 neurons, ReLU activation)
    ↓
Hidden Layer 2 (32 neurons, ReLU activation)
    ↓
Output Layer (1 neuron, Sigmoid activation)
    ↓
Churn Probability (0-1)
```

### Why This Architecture?
- **64 neurons**: Enough capacity to learn complex patterns
- **32 neurons**: Gradually reduces complexity, preventing overfitting
- **2 hidden layers**: Balance between learning capacity and computational efficiency
- **ReLU activation**: Helps with vanishing gradient problem
- **Sigmoid output**: Produces probability scores between 0 and 1

## 📁 Project Structure

```
annclassification/
│
├── 📊 Data
│   └── Churn_Modelling.csv          # Original dataset
│
├── 📓 Notebooks
│   ├── experiments.ipynb            # Main model development
│   ├── hyperparametertuningann.ipynb # Model optimization
│   ├── prediction.ipynb             # Testing predictions
│   └── salaryregression.ipynb       # Bonus: Regression example
│
├── 🤖 Model Files
│   ├── model.h5                     # Trained neural network
│   ├── scaler.pkl                   # Feature scaling parameters
│   ├── label_encoder_gender.pkl     # Gender encoding parameters
│   └── onehot_encoder_geo.pkl       # Geography encoding parameters
│
├── 🌐 Web Application
│   └── app.py                       # Streamlit web interface
│
├── 📋 Configuration
│   ├── requirements.txt             # Python dependencies
│   └── README.md                    # This documentation
│
├── 📊 Logs
│   └── logs/                        # TensorBoard training logs
│
└── 🐍 Environment
    └── myenv/                       # Virtual environment
```

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone/Download the Project
```bash
# If using git
git clone <repository-url>
cd annclassification

# Or download and extract the ZIP file
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Dependencies Explained:
- **tensorflow**: Deep learning framework for building neural networks
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities (preprocessing, metrics)
- **streamlit**: Web application framework
- **matplotlib**: Data visualization
- **tensorboard**: Model training visualization

## 🚀 How to Run the Project

### Option 1: Web Application (Recommended for Beginners)
```bash
# Make sure you're in the project directory
cd annclassification

# Run the Streamlit app
streamlit run app.py
```
This will open a web browser with an interactive interface where you can:
- Input customer information
- Get real-time churn predictions
- Understand the prediction confidence

### Option 2: Jupyter Notebooks (For Learning)
```bash
# Start Jupyter
jupyter notebook

# Open any of the notebooks:
# - experiments.ipynb (main workflow)
# - hyperparametertuningann.ipynb (advanced optimization)
# - prediction.ipynb (manual predictions)
```

### Option 3: Python Scripts
```bash
# If you want to run individual components
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

## 🔧 Technical Implementation

### 1. Data Preprocessing Pipeline

#### Categorical Encoding:
```python
# Label Encoding for Binary Categories
Gender: Male → 1, Female → 0

# One-Hot Encoding for Multiple Categories
Geography: 
  France → [1, 0, 0]
  Germany → [0, 1, 0]  
  Spain → [0, 0, 1]
```

#### Feature Scaling:
```python
# StandardScaler: (value - mean) / standard_deviation
# Ensures all features have mean=0 and std=1
# Prevents features with larger values from dominating
```

### 2. Neural Network Architecture

#### Mathematical Foundation:
```
For each neuron: output = activation(Σ(weights × inputs) + bias)

Hidden Layer 1: 64 neurons
Hidden Layer 2: 32 neurons
Output Layer: 1 neuron (probability)
```

#### Training Process:
1. **Forward Propagation**: Data flows through network to produce prediction
2. **Loss Calculation**: Compare prediction with actual result (Binary Crossentropy)
3. **Backward Propagation**: Calculate gradients and update weights
4. **Repeat**: Until model converges or reaches maximum epochs

### 3. Model Training Configuration

#### Optimizer: Adam
- Adaptive learning rate algorithm
- Combines benefits of AdaGrad and RMSProp
- Good default choice for most problems

#### Loss Function: Binary Crossentropy
- Appropriate for binary classification problems
- Measures difference between predicted and actual probabilities

#### Metrics: Accuracy
- Percentage of correct predictions
- Easy to interpret for business stakeholders

#### Callbacks:
- **Early Stopping**: Prevents overfitting by stopping when validation loss stops improving
- **TensorBoard**: Visualizes training progress and model performance

## 📈 Model Performance

### Training Results:
- **Training Accuracy**: ~85-87%
- **Validation Accuracy**: ~84-86%
- **Loss**: Binary Crossentropy ~0.35-0.40

### Model Evaluation Metrics:

#### Confusion Matrix Understanding:
```
                 Predicted
Actual    No Churn  Churn
No Churn    TN       FP    ← Type II Error (False Positive)
Churn       FN       TP    ← Type I Error (False Negative)
```

#### Business Impact of Errors:
- **False Positive (FP)**: Predicting churn when customer won't churn
  - Cost: Unnecessary retention efforts
- **False Negative (FN)**: Missing actual churners
  - Cost: Lost customers and revenue

### Hyperparameter Tuning Results:
Through grid search, optimal parameters found:
- **Neurons**: 64 (first layer), 32 (second layer)
- **Layers**: 2 hidden layers
- **Epochs**: ~50-100 (with early stopping)
- **Learning Rate**: 0.01

## 🌐 Web Application Features

### User Interface Components:

1. **Geography Selection**: Dropdown menu with country options
2. **Gender Selection**: Male/Female dropdown
3. **Age Slider**: Interactive range selector (18-92)
4. **Numerical Inputs**: 
   - Credit Score
   - Account Balance
   - Estimated Salary
5. **Categorical Selectors**:
   - Tenure (years as customer)
   - Number of products
   - Credit card ownership
   - Active member status

### Prediction Output:
- **Churn Probability**: Percentage likelihood of customer leaving
- **Decision Threshold**: 50% cutoff for churn/no-churn classification
- **User-Friendly Message**: Clear interpretation of results

### Real-time Processing:
1. User inputs data through web interface
2. Data is preprocessed using saved encoders and scaler
3. Neural network generates prediction
4. Result is displayed with explanation

## 🎓 Learning Outcomes

### Data Science Skills:
- **Data Preprocessing**: Handle real-world messy data
- **Feature Engineering**: Transform raw data into model-ready format
- **Model Selection**: Choose appropriate algorithms for business problems
- **Model Evaluation**: Assess performance using relevant metrics

### Deep Learning Concepts:
- **Neural Network Architecture**: Design effective network structures
- **Activation Functions**: Understand when and why to use different functions
- **Loss Functions**: Choose appropriate loss for different problem types
- **Optimization**: Use gradient descent and advanced optimizers

### Software Engineering:
- **Version Control**: Manage code and model versions
- **Environment Management**: Use virtual environments and dependencies
- **Model Deployment**: Create user-friendly applications
- **Documentation**: Write clear, comprehensive documentation

### Business Understanding:
- **Problem Formulation**: Translate business needs into ML problems
- **ROI Calculation**: Understand cost-benefit of ML solutions
- **Stakeholder Communication**: Explain technical concepts to non-technical audiences

## 🚀 Advanced Topics

### 1. Model Improvement Strategies

#### Feature Engineering:
```python
# Create new features from existing ones
customer_value = balance * num_products
engagement_score = is_active_member * tenure
```

#### Ensemble Methods:
- Combine multiple models for better performance
- Use voting classifiers or stacking

#### Advanced Architectures:
- Experiment with different layer sizes
- Try dropout layers for regularization
- Use batch normalization for stable training

### 2. Production Deployment

#### Model Monitoring:
- Track prediction accuracy over time
- Monitor for data drift
- Set up alerts for performance degradation

#### A/B Testing:
- Compare model performance against business rules
- Test different model versions with real customers

#### Scalability:
- Use cloud services (AWS, GCP, Azure)
- Implement batch prediction pipelines
- Set up real-time prediction APIs

### 3. Ethical Considerations

#### Bias Detection:
- Check for discriminatory patterns across demographics
- Ensure fair treatment of all customer segments

#### Privacy Protection:
- Implement data anonymization
- Follow GDPR and other privacy regulations

#### Transparency:
- Provide model explanations to customers
- Document decision-making processes

## 🔍 Troubleshooting

### Common Issues and Solutions:

#### 1. Import Errors
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

#### 2. Model Loading Errors
```bash
# Solution: Ensure all pickle files are in the correct directory
# Re-run the training notebook if files are missing
```

#### 3. Streamlit Issues
```bash
# Solution: Update Streamlit
pip install --upgrade streamlit
```

#### 4. Memory Issues
```bash
# Solution: Reduce batch size or use smaller model
# Monitor system resources during training
```

## 📚 Further Learning

### Recommended Next Steps:
1. **Experiment with Different Architectures**: Try deeper networks, different activation functions
2. **Advanced Preprocessing**: Feature selection, dimensionality reduction
3. **Other ML Algorithms**: Compare with Random Forest, SVM, XGBoost
4. **Time Series Analysis**: Predict when customers will churn
5. **Recommendation Systems**: Suggest products to prevent churn

### Additional Resources:
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Keras Examples](https://keras.io/examples/)

## 🤝 Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting improvements
- Adding new features
- Improving documentation

## 📄 License

This project is for educational purposes. Feel free to use and modify for learning.

---

**Happy Learning! 🎉**

*This project demonstrates the complete machine learning pipeline from data preprocessing to model deployment. Use it as a foundation for your own ML projects and continue exploring the fascinating world of artificial intelligence!*
