# 💳 Credit Scoring Model

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

A beginner-friendly, end-to-end machine learning project that predicts whether a customer is a **Good (Low Risk)** or **Bad (High Risk)** credit customer, based on basic financial information such as income, debt, loans, and payment history.

This project was built to be simple, well-commented, and easy to explain — perfect for learning the full ML workflow, for a portfolio, or for discussing in interviews.

---

## Project Objective

Banks and lenders need a quick way to estimate how risky it is to lend money to a customer. This project simulates that process using a classic **supervised classification** approach:

1. Explore and clean historical credit data
2. Train and compare three classification algorithms
3. Pick the best-performing model
4. Deploy it inside an interactive web app where anyone can enter a customer's details and instantly get a risk prediction

---

## Project Structure

```
credit-scoring-model/
├── data/
│   ├── credit_data.csv          # The dataset used for training (2,000+ records)
│   └── generate_dataset.py      # Script that created the synthetic dataset
├── notebooks/
│   └── EDA.ipynb                # Full exploratory data analysis, with charts
├── models/
│   ├── best_credit_model.pkl    # The saved, best-performing trained model
│   ├── scaler.pkl               # StandardScaler fitted on the training data
│   ├── payment_history_encoder.pkl  # LabelEncoder for the Payment_History column
│   ├── feature_columns.pkl      # The exact feature order the model expects
│   ├── best_model_name.txt      # Name of the best model (used by the app)
│   └── model_metrics.csv        # Accuracy / Precision / Recall / F1 for all 3 models
├── app/
│   └── app.py                   # The Streamlit web application
├── images/
│   └── *.png                    # All charts generated during EDA & evaluation
├── train_model.py               # Main script: cleans data, trains & evaluates models, saves the best one
├── requirements.txt             # All Python dependencies
└── README.md                    # This file
```

---

## About the Dataset

The dataset (`data/credit_data.csv`) contains **2,000+ synthetic customer records** generated to realistically resemble real-world credit data (similar in spirit to the well-known UCI "German Credit Data" dataset, but self-contained so the project runs anywhere with no downloads needed). It was generated with `data/generate_dataset.py`, using logical rules real lenders consider (e.g., high debt-to-income ratio and poor repayment history both increase risk), plus randomness to keep things realistic. A few missing values and duplicate rows were intentionally added so the project includes real data-cleaning practice.

| Column | Description | Type |
|---|---|---|
| `Age` | Customer's age in years | Numerical |
| `Annual_Income` | Customer's yearly income ($) | Numerical |
| `Debt_Amount` | Customer's total existing debt ($) | Numerical |
| `Loan_Amount` | Amount of the loan being requested ($) | Numerical |
| `Number_of_Loans` | Number of loans the customer already holds | Numerical |
| `Payment_History` | Past repayment behavior: Poor / Average / Good | Categorical |
| `Credit_Risk` | **Target** — Good (Low Risk) or Bad (High Risk) | Categorical |

---

## Tech Stack

- **Python 3**
- **Pandas** & **NumPy** — data loading and manipulation
- **Matplotlib** & **Seaborn** — data visualization
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Joblib** — saving/loading the trained model
- **Streamlit** — interactive web app for predictions

---
**Launch the web app**
```bash
streamlit run app/app.py
```
Then open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`).

---

## Machine Learning Workflow

`train_model.py` follows a clear, beginner-friendly pipeline:

1. **Load** the dataset with Pandas
2. **Clean** it — fill missing values (median for numbers, mode for categories) and drop duplicate rows
3. **Encode** the categorical `Payment_History` column with `LabelEncoder`
4. **Split** features (`X`) and target (`y`), then split into 80% train / 20% test sets
5. **Scale** all numerical features with `StandardScaler`
6. **Train** three classification models:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
7. **Evaluate** each model with Accuracy, Precision, Recall, F1-Score, a Confusion Matrix, and a Classification Report
8. **Compare** all three models and automatically select the one with the highest accuracy
9. **Visualize** confusion matrices and Random Forest feature importance
10. **Save** the best model (plus the scaler and encoder it depends on) with `joblib`, ready for the Streamlit app

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 79.5% | 71.0% | 70.0% | 70.5% |
| Decision Tree | 77.0% | 71.4% | 57.1% | 63.5% |
| **Random Forest (Best ✅)** | **84.5%** | **82.5%** | **70.7%** | **76.2%** |

**Random Forest** was automatically selected as the best model based on test-set accuracy, and is the model saved and used by the Streamlit app.

*(Exact numbers will reproduce exactly as above since the dataset and random seeds are fixed — re-running `train_model.py` will not change them unless the data or code is modified.)*

---

## The Streamlit App

`app/app.py` is a clean, professional-looking web interface where a user can enter:

- Age
- Annual Income
- Total Debt Amount
- Loan Amount Requested
- Number of Existing Loans
- Payment History (Poor / Average / Good)

After clicking **Predict Credit Risk**, the app loads the saved model, scaler, and encoder, processes the input the same way it was processed during training, and clearly displays whether the customer is predicted to be **Low Risk ✅** or **High Risk ⚠️**, along with the model's confidence percentage.

---

## Key Skills Demonstrated

This project is a good talking point for interviews because it touches the entire ML lifecycle:

- Exploratory Data Analysis (EDA) and data visualization
- Data cleaning (missing values & duplicates)
- Categorical encoding (Label Encoding)
- Feature scaling (StandardScaler)
- Training & comparing multiple classification algorithms
- Model evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Model persistence with Joblib
- Building and deploying an interactive ML web app with Streamlit
- Clean, modular project structure suitable for real-world / team projects

---

## Future Improvements

- Replace the synthetic dataset with a real-world credit dataset (e.g., UCI German Credit Data)
- Try One-Hot Encoding and compare results against Label Encoding
- Add hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Try additional models such as Gradient Boosting or XGBoost
- Add cross-validation for more robust evaluation
- Deploy the Streamlit app publicly (e.g., Streamlit Community Cloud)

---
