"""
train_model.py

    1. Load the dataset
    2. Clean it (missing values + duplicates)
    3. Encode categorical columns
    4. Split into train/test sets
    5. Scale numerical features
    6. Train 3 beginner-friendly classification models
    7. Evaluate and compare the models
    8. Visualize confusion matrices and feature importance
    9. Save the best model (+ the scaler & encoder it needs) with joblib
"""

import warnings
warnings.filterwarnings("ignore") 

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sns.set_style("whitegrid")


# LOAD THE DATASET
print("STEP 1: Loading dataset...")
df = pd.read_csv("data/credit_data.csv")
print(f"Dataset loaded successfully. Shape: {df.shape}\n")


# CLEAN THE DATASET
print("STEP 2: Cleaning dataset...")


numerical_cols = ["Annual_Income", "Debt_Amount"]
for col in numerical_cols:
    median_value = df[col].median()
    df[col] = df[col].fillna(median_value)


df["Payment_History"] = df["Payment_History"].fillna(df["Payment_History"].mode()[0])

print("Missing values after cleaning:")
print(df.isnull().sum())

duplicates_found = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\nDuplicate rows removed: {duplicates_found}")
print(f"Final cleaned dataset shape: {df.shape}\n")



# ENCODE CATEGORICAL COLUMNSg
print("STEP 3: Encoding categorical columns...")

payment_history_encoder = LabelEncoder()
df["Payment_History"] = payment_history_encoder.fit_transform(df["Payment_History"])
encoding_map = {cls: int(code) for cls, code in
                zip(payment_history_encoder.classes_,
                    payment_history_encoder.transform(payment_history_encoder.classes_))}
print("Payment_History categories encoded as:", encoding_map)


df["Credit_Risk"] = df["Credit_Risk"].map({"Good": 0, "Bad": 1})
print("Credit_Risk encoded as: Good -> 0 (Low Risk), Bad -> 1 (High Risk)\n")


#SEPARATE FEATURES (X) AND TARGET (y)
print("STEP 4: Separating features and target...")
feature_columns = [
    "Age",
    "Annual_Income",
    "Debt_Amount",
    "Loan_Amount",
    "Number_of_Loans",
    "Payment_History",
]
X = df[feature_columns]
y = df["Credit_Risk"]
print(f"Features used: {feature_columns}\n")

# TRAIN/TEST SPLIT (80:20)
print("STEP 5: Splitting data into train and test sets (80:20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}\n")

#SCALE NUMERICAL FEATURES
print("STEP 6: Scaling features with StandardScaler...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.\n")

#TRAIN THE THREE MODELS
print("STEP 7: Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8),
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f" - {name} trained.")
print()


#EVALUATE EACH MODEL
print("STEP 8: Evaluating models...\n")

results = []
confusion_matrices = {}

for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    confusion_matrices[name] = cm
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    })

    print(f"--- {name} ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Good (Low Risk)", "Bad (High Risk)"]))
    print()

results_df = pd.DataFrame(results).set_index("Model")
print("Model Comparison Table:")
print(results_df.round(4))
print()

# VISUALIZE CONFUSION MATRICES
print("STEP 9: Saving confusion matrix plots...")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
labels = ["Good", "Bad"]

for ax, (name, cm) in zip(axes, confusion_matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

plt.tight_layout()
plt.savefig("images/confusion_matrices.png", dpi=120)
plt.close()
print("Saved -> images/confusion_matrices.png\n")

#VISUALIZE MODEL COMPARISON
print("STEP 10: Saving model comparison chart...")
results_df[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(
    kind="bar", figsize=(9, 5), colormap="viridis"
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/model_comparison.png", dpi=120)
plt.close()
print("Saved -> images/model_comparison.png\n")

best_model_name = results_df["Accuracy"].idxmax()
best_model = trained_models[best_model_name]
print(f"STEP 11: Best performing model -> {best_model_name} "
      f"(Accuracy: {results_df.loc[best_model_name, 'Accuracy']:.4f})\n")

#FEATURE IMPORTANCE (RANDOM FOREST)
print("STEP 12: Saving Random Forest feature importance chart...")
rf_model = trained_models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=feature_columns)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh", color="seagreen")
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=120)
plt.close()
print("Saved -> images/feature_importance.png\n")

#SAVE THE BEST MODEL AND PREPROCESSING OBJECTS
print("STEP 13: Saving model and preprocessing objects with joblib...")
joblib.dump(best_model, "models/best_credit_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(payment_history_encoder, "models/payment_history_encoder.pkl")
joblib.dump(feature_columns, "models/feature_columns.pkl")

with open("models/best_model_name.txt", "w") as f:
    f.write(best_model_name)
results_df.round(4).to_csv("models/model_metrics.csv")

print("Saved files:")
print(" - models/best_credit_model.pkl")
print(" - models/scaler.pkl")
print(" - models/payment_history_encoder.pkl")
print(" - models/feature_columns.pkl")
print(" - models/best_model_name.txt")
print(" - models/model_metrics.csv")
print(f"\nTraining complete! Best model ({best_model_name}) is ready for the Streamlit app.")