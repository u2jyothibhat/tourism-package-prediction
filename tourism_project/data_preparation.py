#!/usr/bin/env python3
"""
Data Preparation Script for Tourism Package Prediction Project

This script loads data from HuggingFace, performs cleaning and preprocessing,
splits into train/test sets, and uploads the processed datasets back to HuggingFace.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datasets import Dataset, load_dataset
from huggingface_hub import login
import os
import sys

HF_TOKEN = os.getenv('HF_TOKEN')
login(token=HF_TOKEN)


def load_data_from_hf():
    """Load dataset from Hugging Face Hub"""
    try:
        print("Loading dataset from Hugging Face Hub...")
        dataset = load_dataset("abhishek-kumar/tourism-package-prediction", split="train")
        df = dataset.to_pandas()
        print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Trying to load from local file...")
        try:
            df = pd.read_csv("../tourism.csv")
            print(f"Local dataset loaded: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e2:
            print(f"Error loading local file: {e2}")
            return None

def clean_and_preprocess_data(df):
    """Clean and preprocess the tourism dataset"""
    print("Starting data cleaning and preprocessing...")
    
    # Create a copy
    df_clean = df.copy()
    
    # Remove unnecessary columns (index column if exists)
    if 'Unnamed: 0' in df_clean.columns:
        df_clean = df_clean.drop('Unnamed: 0', axis=1)
    
    print(f"Initial dataset shape: {df_clean.shape}")
    
    # Handle missing values
    print("Missing values before cleaning:")
    missing_before = df_clean.isnull().sum()
    print(missing_before[missing_before > 0])
    
    # Fill missing values with appropriate strategies
    # Numerical columns - fill with median
    numerical_cols = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar', 
                     'NumberOfTrips', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome']
    
    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Categorical columns - fill with mode
    categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus', 
                       'ProductPitched', 'Designation']
    
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    
    print("Missing values after cleaning:")
    missing_after = df_clean.isnull().sum()
    print(missing_after[missing_after > 0])
    
    # Handle data inconsistencies
    # Fix gender inconsistency (Fe Male -> Female)
    if 'Gender' in df_clean.columns:
        df_clean['Gender'] = df_clean['Gender'].replace('Fe Male', 'Female')
    
    # Create feature engineering
    print("Feature engineering...")
    
    # Create income categories
    if 'MonthlyIncome' in df_clean.columns:
        df_clean['IncomeCategory'] = pd.cut(df_clean['MonthlyIncome'], 
                                          bins=[0, 15000, 25000, 35000, float('inf')], 
                                          labels=[0, 1, 2, 3])  # Use numeric labels
    
    # Create age groups
    if 'Age' in df_clean.columns:
        df_clean['AgeGroup'] = pd.cut(df_clean['Age'], 
                                    bins=[0, 25, 35, 45, 55, float('inf')], 
                                    labels=[0, 1, 2, 3, 4])  # Use numeric labels
    
    # Encode categorical variables
    print("Encoding categorical variables...")
    label_encoders = {}
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col != 'CustomerID':  # Don't encode CustomerID
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    print(f"Data preprocessing completed!")
    print(f"Final dataset shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
    
    return df_clean, label_encoders

def split_and_save_data(df):
    """Split data into train/test sets and save locally"""
    print("✂️ Splitting data into train/test sets...")
    
    # Prepare features and target
    X = df.drop(['CustomerID', 'ProdTaken'], axis=1)
    y = df['ProdTaken']
    
    # Split the data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create train and test dataframes
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Add CustomerID back (create new IDs for splits)
    train_df.insert(0, 'CustomerID', range(300000, 300000 + len(train_df)))
    test_df.insert(0, 'CustomerID', range(400000, 400000 + len(test_df)))
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Save locally
    train_path = "data/train_data.csv"
    test_path = "data/test_data.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Data split completed!")
    print(f"   Training set: {len(train_df)} samples")
    print(f"   Test set: {len(test_df)} samples")
    print(f"   Files saved: {train_path}, {test_path}")
    
    return train_df, test_df

def upload_processed_datasets():
    """Upload processed train/test datasets to Hugging Face Hub"""
    print("Uploading processed datasets to Hugging Face Hub...")
    
    try:
        # Load train and test data
        train_df = pd.read_csv("data/train_data.csv")
        test_df = pd.read_csv("data/test_data.csv")
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Upload train dataset
        train_dataset_name = "abhishek-kumar/tourism-package-prediction-train"
        train_dataset.push_to_hub(
            train_dataset_name,
            private=False,
            token=HF_TOKEN
        )
        print(f"Train dataset uploaded: {train_dataset_name}")
        
        # Upload test dataset
        test_dataset_name = "abhishek-kumar/tourism-package-prediction-test"
        test_dataset.push_to_hub(
            test_dataset_name,
            private=False,
            token=HF_TOKEN
        )
        print(f"Test dataset uploaded: {test_dataset_name}")
        
        return True
        
    except Exception as e:
        print(f"Error uploading processed datasets: {e}")
        return False

    
# Load data
df = load_data_from_hf()

# Clean and preprocess
df_clean, encoders = clean_and_preprocess_data(df)

# Split and save
train_df, test_df = split_and_save_data(df_clean)

# Upload processed datasets
upload_processed_datasets()

