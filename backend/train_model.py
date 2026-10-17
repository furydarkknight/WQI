"""
Water Quality Prediction Model Training Script
Based on research: "Data driven water quality assessment using machine learning 
and synthetic data generation" (Scientific Reports, 2026)

Uses: Random Forest, XGBoost, Gradient Boosting, Logistic Regression
with SMOTE oversampling (as recommended by research - best MMD score 0.0067)
Best model: XGBoost/Gradient Boosting (99.47% accuracy on balanced dataset)
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix, brier_score_loss)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not available, using GradientBoosting as primary model")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("imbalanced-learn not available, training on original dataset")


def load_and_preprocess_data(filepath):
    """Load and preprocess the water potability dataset"""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Class distribution:\n{df['Potability'].value_counts()}")
    
    # Features and target
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    # Handle missing values using mean imputation (as per paper)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Min-max normalization (as per paper - Eq. 1)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, imputer, scaler


def train_models(X_train, y_train, X_test, y_test):
    """Train all four models as per the research paper"""
    models = {}
    results = {}
    
    # 1. Logistic Regression
    print("\n--- Training Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # 2. Random Forest  
    print("--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 3. Gradient Boosting
    print("--- Training Gradient Boosting ---")
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, 
                                     max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = gb
    
    # 4. XGBoost (best performer in research)
    if HAS_XGB:
        print("--- Training XGBoost ---")
        xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5,
                            use_label_encoder=False, eval_metric='logloss',
                            random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb
    
    # Evaluate all models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'accuracy': round(acc, 2),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'auc_roc': round(auc, 4),
            'brier_score': round(brier, 4),
            'confusion_matrix': cm.tolist()
        }
        print(f"{name}: Accuracy={acc:.2f}%, AUC={auc:.4f}, F1={f1:.4f}")
    
    return models, results


def save_artifacts(best_model, imputer, scaler, feature_names, results, model_dir):
    """Save model artifacts for the Flask API"""
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the best model
    with open(os.path.join(model_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save imputer
    with open(os.path.join(model_dir, 'imputer.pkl'), 'wb') as f:
        pickle.dump(imputer, f)
    
    # Save scaler
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open(os.path.join(model_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    
    # Save all model results
    with open(os.path.join(model_dir, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nArtifacts saved to {model_dir}")


def main():
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'water_potability.csv')
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    print("=" * 60)
    print("WATER QUALITY PREDICTION MODEL TRAINING")
    print("Based on: Scientific Reports 2026 - Data driven WQA")
    print("=" * 60)
    
    # Load and preprocess
    X, y, imputer, scaler = load_and_preprocess_data(data_path)
    feature_names = list(X.columns)
    
    # Stratified train-test split (80-20 as per paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    
    X_train_final = X_train.copy()
    y_train_final = y_train.copy()
    
    # Apply SMOTE for class balancing (as per paper - best technique)
    if HAS_SMOTE:
        print("\n--- Applying SMOTE oversampling (as per paper) ---")
        print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
        smote = SMOTE(random_state=42)
        X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {pd.Series(y_train_final).value_counts().to_dict()}")
    
    # Train all models
    models, results = train_models(X_train_final, y_train_final, X_test, y_test)
    
    # Select best model (XGBoost > GB > RF > LR as per paper)
    if HAS_XGB and 'XGBoost' in models:
        best_model_name = 'XGBoost'
    else:
        best_model_name = 'Gradient Boosting'
    
    # Override with actual best by AUC
    best_auc = 0
    for name, res in results.items():
        if res['auc_roc'] > best_auc:
            best_auc = res['auc_roc']
            best_model_name = name
    
    print(f"\n[BEST] Best model: {best_model_name} (AUC={best_auc:.4f})")
    
    # 10-fold cross validation on best model (as per paper)
    print(f"\n--- 10-Fold Cross Validation on {best_model_name} ---")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(models[best_model_name], X_train_final, 
                                y_train_final, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    
    results['best_model_name'] = best_model_name
    results['cv_mean'] = round(float(cv_scores.mean() * 100), 2)
    results['cv_std'] = round(float(cv_scores.std() * 100), 2)
    
    # Feature importance
    best_model = models[best_model_name]
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feat_imp = dict(zip(feature_names, [round(float(v), 4) for v in importances]))
        feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
        results['feature_importance'] = feat_imp_sorted
        print("\nFeature Importances:")
        for feat, imp in feat_imp_sorted.items():
            print(f"  {feat}: {imp:.4f}")
    
    # Save all artifacts
    save_artifacts(best_model, imputer, scaler, feature_names, results, model_dir)
    print("\n[DONE] Training complete!")
    print(f"Best Model: {best_model_name}")
    print(f"Test Accuracy: {results[best_model_name]['accuracy']}%")
    print(f"AUC-ROC: {results[best_model_name]['auc_roc']}")


if __name__ == '__main__':
    main()
