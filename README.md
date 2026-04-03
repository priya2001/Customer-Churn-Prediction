# Customer Churn Prediction 

## Project Overview

This project builds a **Machine Learning model** to predict whether a credit card customer will churn (leave the bank) or not. Using a neural network deep learning approach, the model analyzes customer behavior patterns and identifies potential churners before they leave.

**Real-world Application:** Banks can proactively retain at-risk customers through targeted retention strategies.

---

## Objective

Build a predictive model that:
- Identifies customers likely to churn
- Achieves high accuracy in predictions
- Provides actionable insights for customer retention
- Enables data-driven decision making

---

## Dataset

**Source:** [Kaggle - Credit Card Customer Churn Prediction](https://www.kaggle.com/datasets/rjmanoj/credit-card-customer-churn-prediction)

**Dataset Name:** Churn_Modelling.csv

### Dataset Characteristics:
- **Total Records:** 10,000 customers
- **Features:** 13 input features + 1 target variable
- **Target Variable:** `Exited` (0 = No Churn, 1 = Churn)

### Key Features:
| Feature | Description | Data Type |
|---------|-------------|-----------|
| CustomerId | Unique customer identifier | Numeric |
| CreditScore | Customer's credit score | Numeric |
| Geography | Country (France, Germany, Spain) | Categorical |
| Gender | Male/Female | Categorical |
| Age | Customer age | Numeric |
| Tenure | Years as customer | Numeric |
| Balance | Account balance | Numeric |
| NumOfProducts | Number of products used | Numeric |
| HasCrCard | Has credit card (1/0) | Binary |
| IsActiveMember | Active member status (1/0) | Binary |
| EstimatedSalary | Estimated annual salary | Numeric |
| Exited | Churned or not (Target) | Binary |

---

## Project Workflow

```
1. DATA LOADING
   └─ Download dataset from Kaggle
   
2. DATA EXPLORATION
   ├─ Check shape and structure (df.info())
   ├─ Identify duplicates
   ├─ Analyze target distribution (Exited values)
   └─ Check categorical features distribution
   
3. DATA CLEANING
   ├─ Remove unnecessary columns
   │  └─ RowNumber, CustomerId, Surname (not useful)
   ├─ Check for missing values
   └─ Remove duplicates if any
   
4. FEATURE ENGINEERING
   ├─ One-Hot Encoding for categorical variables
   │  ├─ Geography: France, Germany, Spain
   │  └─ Gender: Male, Female
   └─ Keep numerical features as-is
   
5. DATA SCALING
   ├─ Use StandardScaler
   ├─ Fit on training data
   ├─ Transform test data
   └─ Scale all features to mean=0, std=1
   
6. TRAIN-TEST SPLIT
   ├─ Training Set: 80% (8000 samples)
   └─ Testing Set: 20% (2000 samples)
   
7. MODEL BUILDING
   ├─ Create Sequential Neural Network
   ├─ Layer 1: Dense(11) + ReLU
   ├─ Layer 2: Dense(11) + ReLU
   └─ Layer 3: Dense(1) + Sigmoid
   
8. COMPILATION
   ├─ Optimizer: Adam
   ├─ Loss: Binary Crossentropy
   └─ Metrics: Accuracy
   
9. TRAINING
   ├─ Epochs: 100
   ├─ Validation Split: 20%
   └─ Track loss and accuracy
   
10. PREDICTION
    ├─ Predict on test set
    └─ Get probability scores (0-1)
    
11. EVALUATION
    ├─ Apply threshold: 0.5
    ├─ Calculate accuracy
    └─ Visualize loss & accuracy graphs
```

---

## Technologies & Libraries

### **Core Libraries:**
- **Python 3.x** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning tools
- **TensorFlow & Keras** - Deep learning framework
- **Matplotlib** - Data visualization
- **Kaggle API** - Dataset access

### **Specific Tools Used:**

| Tool | Purpose | Usage |
|------|---------|-------|
| `pandas.get_dummies()` | One-hot encoding | Convert categorical to numerical |
| `StandardScaler` | Feature normalization | Scale features to [-3, +3] range |
| `train_test_split()` | Data partitioning | Split into train (80%) and test (20%) |
| `Sequential()` | Neural network model | Create layered architecture |
| `Dense` | Fully connected layer | Create neural network layers |
| `ReLU activation` | Non-linearity | Learn complex patterns |
| `Sigmoid activation` | Output probability | Map to [0, 1] range |
| `Adam optimizer` | Weight optimization | Intelligent gradient descent |
| `Binary Crossentropy` | Loss function | For binary classification |

---

## Model Architecture

```
INPUT LAYER (11 features)
         ↓
Dense Layer 1: 11 neurons + ReLU activation
         ↓
Dense Layer 2: 11 neurons + ReLU activation
         ↓
OUTPUT LAYER: 1 neuron + Sigmoid activation
         ↓
PREDICTION: Probability (0 to 1)
         ↓
THRESHOLD: 0.5
    ├─ If prob > 0.5 → Churn (1) 
    └─ If prob ≤ 0.5 → No Churn (0) 
```

### Layer Details:
- **Input:** 11 features (after one-hot encoding and dropping unnecessary columns)
- **Hidden Layer 1:** 11 neurons with ReLU activation (learns non-linear patterns)
- **Hidden Layer 2:** 11 neurons with ReLU activation (captures deeper patterns)
- **Output Layer:** 1 neuron with Sigmoid activation (outputs probability 0-1)

---

## Data Preprocessing Steps

### 1. **One-Hot Encoding**
```
Geography: ['France', 'Germany', 'Spain']
    ↓ (drop_first=True)
Geography_Germany: 1/0
Geography_Spain: 1/0

Gender: ['Female', 'Male']
    ↓ (drop_first=True)
Gender_Male: 1/0
```

### 2. **Feature Scaling (StandardScaler)**
```
Before: Age=45, Salary=150000, Score=720
After:  Age=-0.5, Salary=+1.2, Score=-0.3
(All features scaled to mean=0, standard deviation=1)
```

### 3. **Why Scaling Matters:**
- Neural networks perform better with normalized inputs
- Prevents large-value features from dominating
- Accelerates training convergence
- Improves model stability

---

## Training Process

### Hyperparameters:
```python
epochs = 100              # Number of complete passes through training data
batch_size = implicit     # Processed in batches (default 32)
validation_split = 0.2    # 20% of training data for validation
optimizer = 'Adam'        # Adaptive learning rate optimizer
loss = 'binary_crossentropy'  # Loss function for binary classification
```

### Monitoring:
- **Training Loss:** How well model fits training data
- **Validation Loss:** How well model generalizes to unseen data
- **Training Accuracy:** % correct predictions on training data
- **Validation Accuracy:** % correct predictions on validation data

### What to Look For:
```
- Good Model: Training and validation curves follow similar pattern
- Overfitting: Training loss decreases but validation loss increases
Underfitting: Both losses remain high
```

---

## Model Performance

### Evaluation Metrics:
- **Accuracy:** Overall correct predictions
- **Precision:** When model predicts churn, how often is it correct?
- **Recall:** Of all actual churners, how many did model catch?
- **F1-Score:** Balanced measure of precision and recall
- **ROC-AUC:** Model's ability to distinguish between classes

### Expected Performance:
- **Typical Accuracy:** 85-90%
- **Threshold:** 0.5 (can be adjusted based on business needs)

---

## How to Use the Model

### Making Predictions:
```python
# New customer data (after scaling)
new_customer = [[scaled_features]]  # 11 features

# Get probability
probability = model.predict(new_customer)  # e.g., 0.78

# Make decision
if probability > 0.5:
    print("High risk of churn - Initiate retention strategy")
else:
    print("Low churn risk - Continue normal service")
```

---

## Business Insights

### Key Factors Influencing Churn:
1. **Age:** Older customers more likely to churn
2. **Tenure:** Long-term customers are more loyal
3. **Geographic Location:** Some countries have higher churn
4. **Number of Products:** Customers with multiple products stay longer
5. **Account Balance:** Lower balance correlates with higher churn risk

### Retention Strategies:
- **Target at-risk segments** using model predictions
- **Personalized offers** for high-probability churners
- **Improve service quality** in high-churn regions
- **Cross-sell products** to increase customer value
- **Proactive customer engagement** before they decide to leave

---

## Project Files

```
CustomerChurnPrediction.ipynb
├── Cell 1-3: Import libraries & load dataset
├── Cell 4-8: Data exploration & analysis
├── Cell 9-12: Data cleaning
├── Cell 13-15: Feature engineering (one-hot encoding)
├── Cell 16-18: Train-test split & scaling
├── Cell 19-21: Model architecture definition
├── Cell 22-24: Model compilation & training
├── Cell 25-27: Predictions on test data
├── Cell 28-31: Model evaluation & visualization
└── Cell 32: Complete
```

---

## Installation & Setup

### Requirements:
```bash
Python 3.7+
```

### Install Dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib kagglehub
```

### Kaggle API Setup:
1. Go to [Kaggle Settings](https://www.kaggle.com/account)
2. Create API token
3. Place `kaggle.json` in `~/.kaggle/`
4. Run kagglehub command in notebook

---

## How to Run

1. **Open the Notebook:**
   ```bash
   jupyter notebook CustomerChurnPrediction.ipynb
   ```

2. **Run Cells Sequentially:**
   - Start from cell 1
   - Execute each cell in order
   - Wait for training to complete (100 epochs ≈ 1-2 minutes)

3. **View Results:**
   - Check accuracy scores
   - Analyze loss/accuracy graphs
   - Review model coefficients

---

## Results Interpretation

### Loss Graphs:
```
Good Training:
├─ Training Loss: Decreases over epochs
├─ Validation Loss: Decreases or plateaus
└─ Both stay close together

Red Flags:
├─ Overfitting: Val loss increases while train loss decreases
├─ Underfitting: Both losses remain high
└─ Divergence: Losses moving in opposite directions
```

### Accuracy Graphs:
```
Good Training:
├─ Training Accuracy: Increases toward 1.0
├─ Validation Accuracy: Increases and plateaus
└─ Both converge to similar values

Warning Signs:
├─ Large gap: Overfitting (train >> validation)
├─ Both low: Model needs improvement
└─ Fluctuating: Learning rate might be too high
```

---

## Customization & Improvements

### Possible Enhancements:

1. **Model Architecture:**
   - Add more layers for deeper learning
   - Try different neuron counts
   - Experiment with dropout layers

2. **Hyperparameter Tuning:**
   - Adjust learning rate
   - Change batch size
   - Modify activation functions

3. **Class Imbalance:**
   - Use class weights
   - Apply SMOTE (Synthetic Minority Oversampling)
   - Adjust decision threshold

4. **Feature Engineering:**
   - Create interaction features
   - Apply polynomial features
   - Domain-specific feature creation

5. **Alternative Models:**
   - Random Forest Classifier
   - XGBoost
   - Support Vector Machines (SVM)
   - Ensemble methods

---

## Learning Resources

- **TensorFlow/Keras:** https://www.tensorflow.org/guide
- **Scikit-learn:** https://scikit-learn.org/
- **Pandas Documentation:** https://pandas.pydata.org/docs
- **Neural Networks:** https://www.deeplearningbook.org/

---

## Contributing

Feel free to fork, modify, and improve this project!

---

## License

This project uses the Kaggle public dataset. Refer to Kaggle's usage terms for more details.

---

## Questions & Support

For questions about the model, refer to:
- Project notebook comments
- Dataset documentation on Kaggle
- TensorFlow/Keras official documentation

---
**Status:** Complete and Ready to Use
