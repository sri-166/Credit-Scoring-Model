import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 2000

# 1. Generate individual features

age = np.random.randint(21, 66, n_samples)

annual_income = np.random.normal(loc=55000, scale=20000, size=n_samples)
annual_income = np.clip(annual_income, 15000, 150000).round(2)

number_of_loans = np.random.randint(0, 9, n_samples)

debt_amount = annual_income * np.random.uniform(0.05, 0.70, n_samples)
debt_amount = debt_amount.round(2)

loan_amount = (number_of_loans + 1) * np.random.uniform(1000, 7000, n_samples)
loan_amount = loan_amount.round(2)

payment_history = np.random.choice(
    ["Poor", "Average", "Good"],
    size=n_samples,
    p=[0.25, 0.35, 0.40]
)

# 2. Build a "risk score" that decides the target label

payment_history_penalty = pd.Series(payment_history).map(
    {"Poor": 25, "Average": 10, "Good": 0}
).values

debt_to_income_ratio = debt_amount / annual_income
loan_to_income_ratio = loan_amount / annual_income

risk_score = (
    (debt_to_income_ratio * 50)
    + (loan_to_income_ratio * 30)
    + (number_of_loans * 2)
    + payment_history_penalty
    - (age - 21) * 0.3 
)

risk_score = risk_score + np.random.normal(0, 8, n_samples)

threshold = np.percentile(risk_score, 65)
credit_risk = np.where(risk_score > threshold, "Bad", "Good")

# 3. Assemble the final DataFrame
df = pd.DataFrame({
    "Age": age,
    "Annual_Income": annual_income,
    "Debt_Amount": debt_amount,
    "Loan_Amount": loan_amount,
    "Number_of_Loans": number_of_loans,
    "Payment_History": payment_history,
    "Credit_Risk": credit_risk
})

# 4. Inject a few missing values (for realistic EDA practice)
for col in ["Annual_Income", "Debt_Amount", "Payment_History"]:
    missing_idx = np.random.choice(df.index, size=15, replace=False)
    df.loc[missing_idx, col] = np.nan

# 5. Inject a few duplicate rows (for realistic cleaning practice)
duplicate_rows = df.sample(10, random_state=42)
df = pd.concat([df, duplicate_rows], ignore_index=True)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/credit_data.csv", index=False)

print("Dataset created successfully!")
print(f"Shape: {df.shape}")
print(df.head())
