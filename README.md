🏦 Loan Approval Prediction – Machine Learning Project


📌 Overview

This project focuses on building a Machine Learning classification model to predict whether a loan application will be approved or rejected based on applicant financial, demographic, and credit-related features.

The goal is to simulate a real-world loan approval system using proper data preprocessing, exploratory data analysis, and model evaluation techniques.



📂 Dataset Description

The dataset contains information such as:

Applicant & Co-applicant Income

Age, Dependents, Marital Status

Employment Status & Employer Category

Credit Score

Debt-to-Income (DTI) Ratio

Existing Loans, Savings

Loan Amount, Loan Term

Target Variable: Loan_Approved



🔍 Exploratory Data Analysis (EDA)

Performed EDA to understand:

Class imbalance in loan approvals

Distribution of numerical features

Relationship between loan approval and:

Income

Credit Score

DTI Ratio

Savings

Feature correlations using heatmaps

Visualizations used:

Histograms

Boxplots

Bar charts

Correlation matrix



🧹 Data Preprocessing

Separated numerical and categorical features

Handled missing values using:

Mean imputation for numerical features

Most frequent imputation for categorical features

Applied One-Hot Encoding for categorical variables

Used StandardScaler for feature scaling

Performed stratified train-test split to handle class imbalance



🤖 Model Building

Algorithm used: Logistic Regression

Reason:

Interpretable

Suitable for binary classification

Strong baseline model



📊 Model Evaluation

Evaluation metrics used:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Classification Report



🔢 Results:

Accuracy: ~80.5%

Precision: ~64.8%

Recall: ~76.7%

F1-Score: ~70.2%

The model performs well in identifying approved loans, with good recall, making it useful in minimizing false rejections.

🛠️ Tech Stack

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Jupyter Notebook
