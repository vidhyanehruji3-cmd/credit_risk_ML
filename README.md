# credit_risk_ML
Project Title
Interpretable Machine Learning: Analyzing Feature Importance and SHAP Values in Credit Risk Prediction

Project Overview
This project focuses on building a credit default classification model using an anonymized or synthetic credit dataset (similar to LendingClub).
Beyond model accuracy, the key objective is to interpret the model using SHAP (SHapley Additive exPlanations) to produce transparent and explainable insights for lending decisions.

The project demonstrates:
Advanced ML model development
Hyperparameter tuning
Global & local interpretability
Clear business recommendations
Analysis of True Positive / False Negative cases

Dataset
A synthetic 2,000-record credit dataset was generated including:

Feature	Description
loan_amnt	Loan amount requested
term_months	Loan term
int_rate	Interest rate
installment	Monthly EMI
annual_inc	Annual income
dti	Debt-to-Income ratio
emp_length	Employment length
home_ownership	Rental/Mortgage/Owned
purpose	Loan purpose
credit_score	Credit rating
revol_bal	Revolving balance
revol_util	Utilization rate
total_acc	Total credit lines
issue_d	Loan issue year
default	Target label

Target variable:
1 = Default, 0 = Non-Default

Project Tasks Completed
1️. Build & Tune Classification Model (XGBoost)
Performed data preprocessing
Encoded categorical variables (OneHot + Ordinal)
Split data into training and test sets
Used GridSearchCV to tune model hyperparameters
Objective optimized for recall due to credit risk business case
Best Model Found:
n_estimators = 80
max_depth = 3
learning_rate = 0.05
2️. Global Feature Importance Analysis
We computed:
✔ Model feature importance
✔ Permutation importance
✔ Global SHAP Values

Top high-impact features discovered:
int_rate
dti
credit_score
annual_inc
revol_util
loan_amnt
These features strongly influence likelihood of loan default.

3️. SHAP Dependence & Interaction Analysis

Generated SHAP plots for 3–5 most important features:
SHAP Summary Plot
SHAP Dependence Plots
SHAP Interaction Plots
Insights include:
Higher interest rate significantly increases default likelihood
Low credit score has strongest positive SHAP value (default risk)
High DTI interacts with income to increase risk

4️. Case Studies (Local SHAP Explanations)

Identified two loans for deep interpretation:
✓ True Positive Case (Correctly Predicted Default)
High interest rate
High DTI
Low credit score
SHAP shows strong positive contributions to default

✓ False Negative Case (Incorrectly Predicted Non-Default)
Model was misled by:
High income
Low utilization
But hidden risk factors included:
Short employment length
High installment burden
SHAP local plots were used to explain misclassification.

5️.Strategic Summary & Business Recommendations
Final written summary includes:
Key findings:
Credit score, DTI, interest rate are primary risk determinants
SHAP allows transparent decision justification
Some misclassifications result from conflicting features

Actionable recommendations:
Implement minimum credit score thresholds
Set stricter limits for DTI > 20%
Reduce loan amounts for high-risk segments
Require additional documentation for applicants with volatile income

Expected Deliverables Completed
✔ Full Python source code: preprocessing, model training, tuning, interpretability
✔ Text-based comparison of feature importance vs. SHAP
✔ Write-up explaining TP/FN case studies
✔ Final business-focused strategic summary

Project Folder Structure
 credit_risk_interpretable_ml
│── README.md
│── data/
│     └── synthetic_credit_data.csv
│── notebooks/
│     └── credit_risk_ml_shap.ipynb
│── src/
│     ├── preprocessing.py
│     ├── model_training.py
│     ├── shap_analysis.py
│     └── utils.py
│── outputs/
│     ├── best_model.pkl
│     ├── feature_importance.png
│     ├── shap_summary.png
│     ├── shap_dependence_plots/
│     ├── case_study_tp.png
│     └── case_study_fn.png

Installation & Dependencies
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn

How to Run
python model_training.py
python shap_analysis.py
Final Notes
This project demonstrates:
Industry-grade interpretability
End-to-end ML pipeline
SHAP-based explainability
Business aligned insights

Perfect for roles in:
FinTech
Credit Risk
Machine Learning Engineering
Data Science
