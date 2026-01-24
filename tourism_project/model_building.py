#!/usr/bin/env python3
"""
Model Building Script for Tourism Package Prediction Project

This script loads processed data, trains multiple ML models with hyperparameter tuning,
tracks experiments with MLflow, and registers the best model to HuggingFace.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datasets import load_dataset
from huggingface_hub import login, HfApi
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

def setup_hf_authentication():
    """Setup Hugging Face authentication"""
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ HF_TOKEN environment variable not found!")
        return False
    
    try:
        login(token=hf_token)
        print("✅ Hugging Face authentication successful!")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def load_train_test_data():
    """Load train and test datasets from Hugging Face Hub"""
    try:
        print("📥 Loading train/test datasets from Hugging Face Hub...")
        
        # Load train dataset
        train_dataset = load_dataset("abhishek-kumar/tourism-package-prediction-train", split="train")
        train_df = train_dataset.to_pandas()
        
        # Load test dataset
        test_dataset = load_dataset("abhishek-kumar/tourism-package-prediction-test", split="train")
        test_df = test_dataset.to_pandas()
        
        print(f"✅ Train dataset loaded: {len(train_df)} samples")
        print(f"✅ Test dataset loaded: {len(test_df)} samples")
        
        return train_df, test_df
        
    except Exception as e:
        print(f"❌ Error loading from HuggingFace: {e}")
        print("📁 Trying to load from local files...")
        try:
            train_df = pd.read_csv("data/train_data.csv")
            test_df = pd.read_csv("data/test_data.csv")
            print(f"✅ Local train dataset loaded: {len(train_df)} samples")
            print(f"✅ Local test dataset loaded: {len(test_df)} samples")
            return train_df, test_df
        except Exception as e2:
            print(f"❌ Error loading local files: {e2}")
            return None, None

def prepare_features(train_df, test_df):
    """Prepare features and target variables"""
    print("🔧 Preparing features and target variables...")
    
    # Remove CustomerID and prepare features
    X_train = train_df.drop(['CustomerID', 'ProdTaken'], axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(['CustomerID', 'ProdTaken'], axis=1)
    y_test = test_df['ProdTaken']
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Feature columns: {list(X_train.columns)}")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"📊 {model_name} Performance:")
    for metric, value in metrics.items():
        print(f"   {metric.capitalize()}: {value:.4f}")
    
    return metrics

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Train Decision Tree with hyperparameter tuning"""
    print("🌳 Training Decision Tree...")
    
    with mlflow.start_run(run_name="DecisionTree"):
        # Define hyperparameters
        param_grid = {
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search
        dt = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "DecisionTree")
        
        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test, "Decision Tree")
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, metrics['roc_auc']

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with hyperparameter tuning"""
    print("🌲 Training Random Forest...")
    
    with mlflow.start_run(run_name="RandomForest"):
        # Define hyperparameters
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Grid search
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "RandomForest")
        
        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test, "Random Forest")
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, metrics['roc_auc']

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting with hyperparameter tuning"""
    print("📈 Training Gradient Boosting...")
    
    with mlflow.start_run(run_name="GradientBoosting"):
        # Define hyperparameters
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7]
        }
        
        # Grid search
        gb = GradientBoostingClassifier(random_state=42)
        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "GradientBoosting")
        
        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test, "Gradient Boosting")
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, metrics['roc_auc']

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with hyperparameter tuning"""
    print("🚀 Training XGBoost...")
    
    with mlflow.start_run(run_name="XGBoost"):
        # Define hyperparameters
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9]
        }
        
        # Grid search
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "XGBoost")
        
        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test, "XGBoost")
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.xgboost.log_model(best_model, "model")
        
        return best_model, metrics['roc_auc']

def train_adaboost(X_train, y_train, X_test, y_test):
    """Train AdaBoost with hyperparameter tuning"""
    print("🎯 Training AdaBoost...")
    
    with mlflow.start_run(run_name="AdaBoost"):
        # Define hyperparameters
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.5, 1.0, 1.5]
        }
        
        # Grid search
        ada = AdaBoostClassifier(random_state=42)
        grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Log parameters
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "AdaBoost")
        
        # Evaluate
        metrics = evaluate_model(best_model, X_test, y_test, "AdaBoost")
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        return best_model, metrics['roc_auc']

def register_best_model(best_model, best_model_name, best_score):
    """Register the best model to HuggingFace Hub"""
    print(f"🏆 Registering best model: {best_model_name} (ROC-AUC: {best_score:.4f})")
    
    try:
        # Create model directory
        os.makedirs("model_building", exist_ok=True)
        
        # Save model locally
        model_path = "model_building/best_model.joblib"
        joblib.dump(best_model, model_path)
        
        # Create model info file
        model_info = {
            "model_name": best_model_name,
            "roc_auc_score": best_score,
            "model_type": "classification",
            "framework": "scikit-learn" if "XG" not in best_model_name else "xgboost"
        }
        
        with open("model_building/model_info.txt", "w") as f:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        
        # Upload to HuggingFace Model Hub
        api = HfApi()
        repo_id = "abhishek-kumar/tourism-package-prediction-model"
        
        # Create or update repository
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        except:
            pass  # Repo might already exist
        
        # Upload model file
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model.joblib",
            repo_id=repo_id,
            token=os.getenv('HF_TOKEN')
        )
        
        # Upload model info
        api.upload_file(
            path_or_fileobj="model_building/model_info.txt",
            path_in_repo="model_info.txt",
            repo_id=repo_id,
            token=os.getenv('HF_TOKEN')
        )
        
        print(f"✅ Best model registered to HuggingFace: {repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return False

def main():
    """Main model building pipeline"""
    print("🚀 Tourism Package Prediction - Model Building Pipeline")
    print("="*60)
    
    # Setup MLflow
    mlflow.set_experiment("tourism_package_prediction")
    
    # Load data
    train_df, test_df = load_train_test_data()
    if train_df is None or test_df is None:
        return False
    
    # Prepare features
    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df)
    
    # Train multiple models
    models_results = []
    
    try:
        # Decision Tree
        dt_model, dt_score = train_decision_tree(X_train, y_train, X_test, y_test)
        models_results.append(("Decision Tree", dt_model, dt_score))
        
        # Random Forest
        rf_model, rf_score = train_random_forest(X_train, y_train, X_test, y_test)
        models_results.append(("Random Forest", rf_model, rf_score))
        
        # Gradient Boosting
        gb_model, gb_score = train_gradient_boosting(X_train, y_train, X_test, y_test)
        models_results.append(("Gradient Boosting", gb_model, gb_score))
        
        # XGBoost
        xgb_model, xgb_score = train_xgboost(X_train, y_train, X_test, y_test)
        models_results.append(("XGBoost", xgb_model, xgb_score))
        
        # AdaBoost
        ada_model, ada_score = train_adaboost(X_train, y_train, X_test, y_test)
        models_results.append(("AdaBoost", ada_model, ada_score))
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False
    
    # Find best model
    best_model_name, best_model, best_score = max(models_results, key=lambda x: x[2])
    
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    for name, model, score in models_results:
        print(f"{name:<20}: ROC-AUC = {score:.4f}")
    
    print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {best_score:.4f})")
    
    # Register best model
    if setup_hf_authentication():
        register_success = register_best_model(best_model, best_model_name, best_score)
        if register_success:
            print("🎉 Model building pipeline completed successfully!")
        else:
            print("⚠️ Model training completed but registration failed")
    else:
        print("⚠️ Model training completed but authentication failed for registration")
    
    return True

def prepare_features(train_df, test_df):
    """Prepare features and target variables"""
    print("🔧 Preparing features and target variables...")
    
    # Remove CustomerID and prepare features
    X_train = train_df.drop(['CustomerID', 'ProdTaken'], axis=1)
    y_train = train_df['ProdTaken']
    X_test = test_df.drop(['CustomerID', 'ProdTaken'], axis=1)
    y_test = test_df['ProdTaken']
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Feature columns: {list(X_train.columns)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 