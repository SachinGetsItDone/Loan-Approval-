import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================================
# Data loading (replicates notebook context)
# =========================================

@st.cache_data
def load_data():
    columns = [
        "Applicant_ID",
        "Applicant_Income",
        "Coapplicant_Income",
        "Employment_Status",
        "Age",
        "Marital_Status",
        "Dependents",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
        "Loan_Purpose",
        "Property_Area",
        "Education_Level",
        "Gender",
        "Employer_Category",
        "Loan_Approved",
    ]

    data = [
        [1.0, 17795.0, 1387.0, "Salaried", 51.0, "Married", 0.0, 637.0, 4.0, 0.53, 19403.0, 45638.0, 16619.0, 84.0,
         "Personal", "Urban", "Not Graduate", "Female", "Private", "No"],
        [2.0, 2860.0, 2679.0, "Salaried", 46.0, "Married", 3.0, 621.0, 2.0, 0.30, 2580.0, 49272.0, 38687.0, np.nan,
         "Car", "Semiurban", "Graduate", np.nan, "Private", "No"],
        [3.0, 7390.0, 2106.0, "Salaried", 25.0, "Single", 2.0, 674.0, 4.0, 0.20, 13844.0, 6908.0, 27943.0, 72.0,
         np.nan, "Urban", np.nan, "Female", "Government", "Yes"],
        [4.0, 13964.0, 8173.0, "Salaried", 40.0, "Married", 2.0, 579.0, 3.0, 0.31, 9553.0, 10844.0, 27819.0, 60.0,
         "Business", "Rural", "Graduate", "Female", "Government", "No"],
        [5.0, 13284.0, 4223.0, "Self-employed", 31.0, "Single", 2.0, 721.0, 1.0, 0.29, 9386.0, 37629.0, 12741.0, 72.0,
         "Car", np.nan, "Graduate", "Male", "Private", "Yes"],
    ]

    return pd.DataFrame(data, columns=columns)


# =========================================
# Training pipeline (replicated)
# =========================================

@st.cache_resource
def train_final_model():
    df = load_data()

    target_col = "Loan_Approved"
    X = df.drop(columns=[target_col])
    y = df[target_col].map({"Yes": 1, "No": 0})

    numeric_features = [
        "Applicant_ID",
        "Applicant_Income",
        "Coapplicant_Income",
        "Age",
        "Dependents",
        "Credit_Score",
        "Existing_Loans",
        "DTI_Ratio",
        "Savings",
        "Collateral_Value",
        "Loan_Amount",
        "Loan_Term",
    ]

    categorical_features = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Education_Level",
        "Gender",
        "Employer_Category",
    ]

    X_num = X[numeric_features].copy()
    for col in numeric_features:
        X_num[col].fillna(X_num[col].median(), inplace=True)

    X_cat = X[categorical_features].copy()
    for col in categorical_features:
        X_cat[col].fillna(X_cat[col].mode(dropna=True)[0], inplace=True)

    X_cat_dummies = pd.get_dummies(X_cat, drop_first=False)
    X_processed = pd.concat([X_num, X_cat_dummies], axis=1)

    scaler = StandardScaler()
    X_processed[numeric_features] = scaler.fit_transform(X_processed[numeric_features])

    model = LogisticRegression(max_iter=1000)
    model.fit(X_processed, y)

    categories = {col: sorted(df[col].dropna().unique()) for col in categorical_features}

    return {
        "model": model,
        "scaler": scaler,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_columns": X_processed.columns.tolist(),
        "categories": categories,
    }


# =========================================
# Prediction helpers
# =========================================

def preprocess_input(user_df, pipeline):
    X_num = user_df[pipeline["numeric_features"]].copy()
    X_cat = user_df[pipeline["categorical_features"]].copy()

    X_cat = pd.get_dummies(X_cat, drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.reindex(columns=pipeline["all_columns"], fill_value=0)
    X[pipeline["numeric_features"]] = pipeline["scaler"].transform(
        X[pipeline["numeric_features"]]
    )
    return X


def predict_approval(user_df, pipeline):
    X = preprocess_input(user_df, pipeline)
    pred = pipeline["model"].predict(X)
    return "Approved" if pred[0] == 1 else "Not Approved"


# =========================================
# Streamlit UI
# =========================================

def main():
    st.title("Loan Approval Prediction")
    pipeline = train_final_model()

    applicant_id = st.number_input("Applicant_ID", value=1.0, step=1.0)
    applicant_income = st.number_input("Applicant_Income", value=5000.0)
    coapplicant_income = st.number_input("Coapplicant_Income", value=0.0)
    age = st.number_input("Age", value=30.0)
    dependents = st.number_input("Dependents", value=0.0)
    credit_score = st.number_input("Credit_Score", value=650.0)
    existing_loans = st.number_input("Existing_Loans", value=0.0)
    dti_ratio = st.number_input("DTI_Ratio", value=0.3)
    savings = st.number_input("Savings", value=10000.0)
    collateral_value = st.number_input("Collateral_Value", value=20000.0)
    loan_amount = st.number_input("Loan_Amount", value=10000.0)
    loan_term = st.number_input("Loan_Term", value=36.0)

    cats = pipeline["categories"]

    employment_status = st.selectbox("Employment_Status", cats["Employment_Status"])
    marital_status = st.selectbox("Marital_Status", cats["Marital_Status"])
    loan_purpose = st.selectbox("Loan_Purpose", cats["Loan_Purpose"])
    property_area = st.selectbox("Property_Area", cats["Property_Area"])
    education_level = st.selectbox("Education_Level", cats["Education_Level"])
    gender = st.selectbox("Gender", cats["Gender"])
    employer_category = st.selectbox("Employer_Category", cats["Employer_Category"])

    input_df = pd.DataFrame({
        "Applicant_ID": [applicant_id],
        "Applicant_Income": [applicant_income],
        "Coapplicant_Income": [coapplicant_income],
        "Employment_Status": [employment_status],
        "Age": [age],
        "Marital_Status": [marital_status],
        "Dependents": [dependents],
        "Credit_Score": [credit_score],
        "Existing_Loans": [existing_loans],
        "DTI_Ratio": [dti_ratio],
        "Savings": [savings],
        "Collateral_Value": [collateral_value],
        "Loan_Amount": [loan_amount],
        "Loan_Term": [loan_term],
        "Loan_Purpose": [loan_purpose],
        "Property_Area": [property_area],
        "Education_Level": [education_level],
        "Gender": [gender],
        "Employer_Category": [employer_category],
    })

    if st.button("Predict Loan Approval"):
        result = predict_approval(input_df, pipeline)
        st.success(f"Loan Status: {result}")


if __name__ == "__main__":
    main()
