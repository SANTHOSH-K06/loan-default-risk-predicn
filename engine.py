import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

class LoanClassifierEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        self.le_mapping = {
            'Male': 1, 'Female': 0,
            'Yes': 1, 'No': 0,
            'Graduate': 1, 'Not Graduate': 0,
            'Urban': 2, 'Semiurban': 1, 'Rural': 0,
            'Y': 1, 'N': 0,
            '0': 0, '1': 1, '2': 2, '3+': 3
        }
        self.feature_names = None

    def load_data(self, file_path='train.csv'):
        try:
            df = pd.read_csv(file_path)
            return df
        except:
            return None

    def preprocess(self, df, is_train=True):
        df = df.copy()
        if 'Loan_ID' in df.columns:
            df.drop('Loan_ID', axis=1, inplace=True)
            
        cat_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0])
        
        num_cols = ['LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome', 'CoapplicantIncome']
        for col in num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Consistent mapping
        for col in df.columns:
            if df[col].dtype == 'object' or col in ['Credit_History', 'Dependents']:
                df[col] = df[col].astype(str).map(lambda x: self.le_mapping.get(x, x))
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        if is_train:
            if 'Loan_Status' in df.columns:
                y = df['Loan_Status']
                X = df.drop('Loan_Status', axis=1)
                self.feature_names = X.columns
                X_scaled = self.scaler.fit_transform(X)
                return X_scaled, y, self.feature_names
        else:
            if self.feature_names is not None:
                # Ensure columns match training
                for col in self.feature_names:
                    if col not in df.columns:
                        df[col] = 0
                df = df[self.feature_names]
            X_scaled = self.scaler.transform(df)
            return X_scaled

    def optimize_and_train(self, X, y, algorithm='LR', C=1.0):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if algorithm == 'LR':
            model = LogisticRegression(C=C, max_iter=1000, random_state=42, solver='liblinear')
            model.fit(X_train, y_train)
        else:
            model = SVC(C=C, probability=True, kernel='rbf', random_state=42)
            model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except:
            auc = 0.0
            
        return model, acc, auc
