# Loan Approval Prediction (Machine Learning)

## Overview
This project builds a **binary classification model** to predict whether a loan application will be **approved or rejected** based on applicant financial, demographic, and credit-related information.

The project follows a complete **end-to-end machine learning workflow**, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

---

## Dataset
The dataset includes the following types of features:
- Applicant & Co-applicant Income
- Age, Dependents, Marital Status
- Employment Status & Employer Category
- Credit Score
- Debt-to-Income (DTI) Ratio
- Existing Loans, Savings
- Loan Amount and Loan Term

**Target Variable:**  
- `Loan_Approved` (0 = No, 1 = Yes)

---

## Exploratory Data Analysis (EDA)
EDA was performed to:
- Analyze class distribution of loan approvals
- Understand distributions of numerical features
- Compare approved vs rejected loans using boxplots
- Study correlations using a correlation heatmap

Visualizations used:
- Histograms
- Boxplots
- Bar charts
- Correlation matrix

---

## Data Preprocessing
- Separated numerical and categorical features
- Handled missing values using:
  - Mean imputation for numerical features
  - Most frequent value imputation for categorical features
- Applied One-Hot Encoding to categorical variables
- Standardized numerical features using `StandardScaler`
- Used stratified train-test split to handle class imbalance

---

## Model
- **Algorithm:** Logistic Regression  
- Chosen as a strong and interpretable baseline model for binary classification

---

## Model Evaluation
Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

### Performance:
- Accuracy: ~80.5%
- Precision: ~64.8%
- Recall: ~76.7%
- F1-Score: ~70.2%

The model shows good recall, making it effective at identifying approved loan applications.

---

## Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- streamlit

---

