
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
n = 800

gender = np.random.choice(['Male', 'Female'], n, p=[0.8, 0.2])
married = np.random.choice(['Yes', 'No'], n, p=[0.65, 0.35])
dependents = np.random.choice([0, 1, 2, 3], n, p=[0.55, 0.2, 0.15, 0.1])
education = np.random.choice(['Graduate', 'Not Graduate'], n, p=[0.78, 0.22])
self_employed = np.random.choice(['Yes', 'No'], n, p=[0.15, 0.85])
applicant_income = np.random.gamma(5, 1200, n).astype(int) + 1500
coapplicant_income = np.random.choice([0, 1], n, p=[0.4, 0.6]) * (np.random.gamma(3, 800, n).astype(int))
loan_amount = (applicant_income + coapplicant_income) / np.random.uniform(15, 40, n)
loan_amount = loan_amount.astype(int)
loan_term = np.random.choice([120, 180, 240, 300, 360], n, p=[0.05, 0.1, 0.1, 0.15, 0.6])
credit_history = np.random.choice([1, 0], n, p=[0.84, 0.16])
property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], n, p=[0.38, 0.38, 0.24])

# Rule-based score to generate a realistic target, then add noise
score = (
    credit_history * 4.0
    + (education == 'Graduate') * 0.6
    + (applicant_income + coapplicant_income > 5000) * 1.0
    + (loan_amount < 150) * 0.8
    + (property_area != 'Rural') * 0.5
    - (dependents >= 3) * 0.4
    + np.random.normal(0, 1.1, n)
)
loan_status = (score > 3.2).astype(int)  # 1 = Approved, 0 = Rejected

df = pd.DataFrame({
    'Gender': gender,
    'Married': married,
    'Dependents': dependents,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_history,
    'Property_Area': property_area,
    'Loan_Status': loan_status
})

# Encode categoricals (simple manual maps, saved for reuse in Flask app)
maps = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
    'Self_Employed': {'Yes': 1, 'No': 0},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
}

X = df.copy()
for col, m in maps.items():
    X[col] = X[col].map(m)

feature_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

X_train, X_test, y_train, y_test = train_test_split(
    X[feature_cols], X['Loan_Status'], test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy: {test_acc:.3f}")

with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'maps': maps, 'features': feature_cols}, f)

print("Saved model.pkl")
