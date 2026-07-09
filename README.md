💳 Smart Lender
A Machine Learning based web application that predicts loan approval status for applicants, helping banks and financial institutions speed up credit decisions

App Link
https://wincing-dropout-grouped.ngrok-free.dev/

🔍 About the Project
Loan approval is traditionally a slow, manual process involving multiple checks on an applicant's financial background. **Smart Lender** automates this by using a trained classification model to instantly predict whether a loan application is likely to be **Approved** or **Rejected**, along with a confidence score — reducing turnaround time for credit officers and analysts.

⚙️ How It Works
1. Applicant details are entered through a web form (income, credit history, employment type, etc.)
2. The input is encoded to match the model's training format
3. A trained **Random Forest Classifier** predicts the loan outcome
4. The result is displayed instantly with a confidence percentage
   
🧰 Tech Stack
Python                          
Flask                           
Scikit-learn                    
Pandas
NumPy                   
HTML5
CSS3 

📂 Repository Structure
Smart-Lender/
├── train_model.py       → builds dataset, trains RandomForest model
├── app.py                → Flask backend serving predictions
├── model.pkl              → trained model + label encodings
├── templates/
│   ├── home.html          → landing page
│   ├── predict.html       → applicant input form
│   └── submit.html        → prediction output page
└── README.md

📝 Fields Collected From Applicants
- Gender
- Marital Status
- Number of Dependents
- Education Level
- Self-Employment Status
- Applicant Income
- Co-applicant Income
- Loan Amount Requested
- Loan Term
- Credit History
- Property Area
  
Getting Started
**Install dependencies:**
```
pip install flask scikit-learn pandas numpy pyngrok
```
**Train the model:**
```
python train_model.py
```
**Run the app locally:**
```
python app.py
```
App runs at `http://127.0.0.1:5000`

💡 Use Cases
- **Quick approvals:** Salaried applicants with strong credit history get instant, high-confidence approvals
- **Risk flagging:** Self-employed applicants with irregular income or no credit history are flagged for manual review
- **Bulk processing:** Analysts can run predictions across many applicants quickly during high application volumes.

