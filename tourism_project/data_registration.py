#!/usr/bin/env python3
"""
Data Registration Script for Tourism Package Prediction Project

This script uploads the original tourism dataset to Hugging Face Hub.
"""

import pandas as pd
from datasets import Dataset
from huggingface_hub import login, HfApi
import os
import sys

def setup_hf_authentication():
    """Setup Hugging Face authentication"""
    print("Setting up Hugging Face authentication...")
    
    # Check if HF_TOKEN environment variable is set
    hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        print("❌ HF_TOKEN environment variable not found!")
        print("\nTo set up authentication:")
        print("1. Go to https://huggingface.co/settings/tokens")
        print("2. Create a new token with 'write' permissions")
        print("3. Set the token as environment variable:")
        print("   export HF_TOKEN='your_token_here'")
        return False
    
    try:
        login(token=hf_token)
        print("✅ Hugging Face authentication successful!")
        return True
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def register_original_dataset():
    """Register the original tourism dataset to Hugging Face Hub"""
    print("📊 Registering Original Tourism Dataset")
    print("="*50)
    
    # Setup authentication
    if not setup_hf_authentication():
        return False
    
    try:
        # Load the original dataset
        dataset_path = "../tourism.csv"
        if not os.path.exists(dataset_path):
            dataset_path = "tourism.csv"  # Try current directory
        
        print(f"Loading dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Display basic info
        print(f"Target variable distribution:")
        print(df['ProdTaken'].value_counts())
        
        # Create HuggingFace dataset
        dataset = Dataset.from_pandas(df)
        
        # Upload to Hugging Face Hub
        dataset_name = "abhishek-kumar/tourism-package-prediction"
        dataset.push_to_hub(
            dataset_name,
            private=False,
            token=os.getenv('HF_TOKEN')
        )
        
        print(f"✅ Successfully uploaded original dataset!")
        print(f"   URL: https://huggingface.co/datasets/{dataset_name}")
        
        # Save dataset info for later use
        with open("data/dataset_info.txt", "w") as f:
            f.write(f"Original Dataset: {dataset_name}\n")
            f.write(f"Rows: {len(df)}\n")
            f.write(f"Columns: {len(df.columns)}\n")
            f.write(f"Target Distribution:\n{df['ProdTaken'].value_counts()}\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        return False

if __name__ == "__main__":
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Register dataset
    success = register_original_dataset()
    sys.exit(0 if success else 1) 