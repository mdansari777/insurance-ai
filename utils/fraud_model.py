import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from .preprocessing import get_fraud_preprocessor

def train_fraud_model():
    print("="*50)
    print("Training Fraud Detection Model...")
    df = pd.read_csv('data/fraud_claims.csv')
    X = df.drop(columns=['fraud_reported'])
    y = df['fraud_reported']
    preprocessor = get_fraud_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    }
    param_grids = {
        'LogisticRegression': {'classifier__C': [0.1,1,10]},
        'RandomForest': {'classifier__n_estimators': [50,100], 'classifier__max_depth': [5,10]},
        'XGBoost': {'classifier__n_estimators': [50,100], 'classifier__learning_rate': [0.05,0.1]}
    }

    best_estimators = {}
    results = []
    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
        param_grid = param_grids[name]
        gs = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_
        y_pred = best_pipe.predict(X_test)
        y_proba = best_pipe.predict_proba(X_test)[:,1]
        
        results.append({
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        })
        best_estimators[name] = best_pipe

    best_model_name = max(results, key=lambda x: x['roc_auc'])['model']
    best_model = best_estimators[best_model_name]

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/fraud_best.pkl')
    pd.DataFrame(results).to_csv('models/fraud_classification_results.csv', index=False)
    print(f"✅ Best fraud model: {best_model_name} (ROC-AUC={max(r['roc_auc'] for r in results):.3f})")

if __name__ == "__main__":
    train_fraud_model()