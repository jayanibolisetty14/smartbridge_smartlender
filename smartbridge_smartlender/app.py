
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    maps = data['maps']
    features = data['features']


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict')
def predict_form():
    return render_template('predict.html')


@app.route('/submit', methods=['POST'])
def submit():
    form = request.form

    gender = maps['Gender'][form.get('gender')]
    married = maps['Married'][form.get('married')]
    dependents = int(form.get('dependents'))
    education = maps['Education'][form.get('education')]
    self_employed = maps['Self_Employed'][form.get('self_employed')]
    applicant_income = float(form.get('applicant_income'))
    coapplicant_income = float(form.get('coapplicant_income') or 0)
    loan_amount = float(form.get('loan_amount'))
    loan_term = float(form.get('loan_term'))
    credit_history = int(form.get('credit_history'))
    property_area = maps['Property_Area'][form.get('property_area')]

    row = np.array([[gender, married, dependents, education, self_employed,
                      applicant_income, coapplicant_income, loan_amount,
                      loan_term, credit_history, property_area]])

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    result = "Approved" if pred == 1 else "Rejected"
    confidence = round((proba if pred == 1 else 1 - proba) * 100, 1)

    return render_template('submit.html', result=result, confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)
