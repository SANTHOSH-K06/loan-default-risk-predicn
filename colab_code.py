# 🏦 Loan Default Risk Prediction - Financial Risk Analytics
# Algorithms: Logistic Regression, SVM
# Optimization: Regularization tuning, GridSearchCV, Cross-Validation

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
# In Colab, you can upload the files or link them from Drive.
# For this script, we assume train.csv and test_Y3wMUE5_7gLdaTN.csv are in the current directory.
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ Files not found. Please upload 'train.csv' and 'test_Y3wMUE5_7gLdaTN.csv' to Colab.")
    # Exit if files are missing
    import sys
    sys.exit()

# --- 2. Data Cleaning & Preprocessing ---
def preprocess_data(df, is_train=True):
    df_copy = df.copy()
    
    # Drop Loan_ID as it's not a feature
    if 'Loan_ID' in df_copy.columns:
        df_copy.drop('Loan_ID', axis=1, inplace=True)
    
    # Handling Missing Values
    # Categorical: Fill with Mode
    cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
    for col in cat_cols:
        if col in df_copy.columns:
            df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
            
    # Numerical: Fill with Median
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for col in num_cols:
        if col in df_copy.columns:
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
    
    # Encoding Categorical Variables
    # Standard mapping for consistent results
    mapping = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Loan_Status': {'Y': 1, 'N': 0}
    }
    
    for col, map_val in mapping.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(map_val)
            
    return df_copy

train_cleaned = preprocess_data(train_df)
test_cleaned = preprocess_data(test_df, is_train=False)

# Define X and y
X = train_cleaned.drop('Loan_Status', axis=1)
y = train_cleaned['Loan_Status']

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_cleaned) if not test_cleaned.empty else None

# --- 3. Logistic Regression Optimization ---
print("\n--- Optimizing Logistic Regression ---")
lr_param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_param_grid, cv=5, scoring='accuracy')
lr_grid.fit(X_train_scaled, y_train)

best_lr = lr_grid.best_estimator_
print(f"Best Parameters: {lr_grid.best_params_}")
print(f"CV Score: {lr_grid.best_score_:.4f}")

# --- 4. SVM Optimization ---
print("\n--- Optimizing SVM ---")
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_param_grid, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)

best_svm = svm_grid.best_estimator_
print(f"Best Parameters: {svm_grid.best_params_}")
print(f"CV Score: {svm_grid.best_score_:.4f}")

# --- 5. Model Evaluation & Comparison ---
def evaluate_model(model, X_v, y_v, name):
    preds = model.predict(X_v)
    proba = model.predict_proba(X_v)[:, 1]
    acc = accuracy_score(y_v, preds)
    auc = roc_auc_score(y_v, proba)
    
    print(f"\n📊 {name} Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_v, preds))
    
    return acc, auc

lr_acc, lr_auc = evaluate_model(best_lr, X_val_scaled, y_val, "Logistic Regression")
svm_acc, svm_auc = evaluate_model(best_svm, X_val_scaled, y_val, "SVM")

# --- 6. Visualization ---
plt.figure(figsize=(12, 5))

# Confusion Matrix for Best Model (let's assume SVM or LR)
best_model = best_svm if svm_acc > lr_acc else best_lr
cm = confusion_matrix(y_val, best_model.predict(X_val_scaled))

plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title(f'Confusion Matrix ({type(best_model).__name__})')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Accuracy Comparison
plt.subplot(1, 2, 2)
models = ['LogReg', 'SVM']
accuracies = [lr_acc, svm_acc]
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.title('Accuracy Comparison')
plt.ylim(0, 1)

plt.show()

# --- 7. Final Predictions on Test Data ---
if X_test_scaled is not None:
    test_preds = best_model.predict(X_test_scaled)
    test_df['Loan_Status_Predicted'] = ['Y' if p == 1 else 'N' for p in test_preds]
    print("\n✅ Predictions for test set completed.")
    print(test_df[['Loan_ID', 'Loan_Status_Predicted']].head())
    # Save results
    test_df.to_csv('loan_predictions_result.csv', index=False)
    print("📂 Results saved to 'loan_predictions_result.csv'")
