# Tourism Package Prediction MLOps Project

## Project Overview

"Visit with Us" tourism company's MLOps pipeline for predicting customer purchase likelihood of the Wellness Tourism Package. This end-to-end machine learning solution automates customer targeting through data-driven predictions.

## Business Objective

Build an automated system that:
- Predicts whether customers will purchase the Wellness Tourism Package
- Optimizes marketing campaigns through targeted customer identification
- Implements scalable MLOps practices for continuous model improvement
- Reduces manual effort and improves campaign performance

## Dataset Description

The dataset contains customer demographics and interaction data with 20 features:

**Customer Details:**
- CustomerID, Age, Gender, MaritalStatus, CityTier, Occupation, Designation
- MonthlyIncome, NumberOfPersonVisiting, NumberOfChildrenVisiting
- NumberOfTrips, Passport, OwnCar, PreferredPropertyStar

**Sales Interaction Data:**
- TypeofContact, DurationOfPitch, ProductPitched, NumberOfFollowups
- PitchSatisfactionScore

**Target Variable:**
- ProdTaken (0: No purchase, 1: Purchase)

## MLOps Pipeline Architecture

### 1. Data Registration
- Upload original dataset to HuggingFace Hub
- Establish data versioning and accessibility

### 2. Data Preparation
- Load data from HuggingFace Hub
- Clean and handle missing values
- Feature engineering (income categories, age groups)
- Encode categorical variables
- Split into train/test sets (80/20)
- Upload processed datasets to HuggingFace

### 3. Model Building & Experimentation
- Train multiple ML algorithms:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - AdaBoost
- Hyperparameter tuning with GridSearchCV
- MLflow experiment tracking
- Model evaluation and comparison
- Register best model to HuggingFace Model Hub

### 4. Deployment
- Containerized deployment with Docker
- Streamlit web application for predictions
- Automated deployment to HuggingFace Spaces

### 5. CI/CD Pipeline
- GitHub Actions workflow automation
- Automated testing and deployment
- Continuous integration on main branch updates

## Project Structure

```
tourism_project/
├── data_registration.py       # Upload original dataset
├── data_preparation.py        # Data cleaning and preprocessing
├── model_building.py         # ML model training and tracking
├── requirements.txt          # Python dependencies
├── data/                     # Data storage
├── model_building/           # Model artifacts
└── deployment/               # Deployment files
    ├── app.py               # Streamlit application
    ├── Dockerfile           # Container configuration
    ├── requirements.txt     # Deployment dependencies
    └── deploy_to_hf.py      # HuggingFace deployment script
```

## Live Links

- **Streamlit App**: https://huggingface.co/spaces/u2jyothibhat/tourism_project
- **Original Dataset**: https://huggingface.co/datasets/u2jyothibhat/tourism_project
- **Train Dataset**: https://huggingface.co/datasets/u2jyothibhat/tourism-package-prediction-train
- **Test Dataset**: https://huggingface.co/datasets/u2jyothibhat/tourism-package-prediction-test
- **Model Hub**: https://huggingface.co/u2jyothibhat/tourism-package-prediction-model

## Setup Instructions

### Prerequisites
- Python 3.9+
- HuggingFace account and token
- GitHub repository with Actions enabled

### Environment Setup
```bash
# Activate conda environment
conda activate applied_stats_project

# Install dependencies
pip install -r tourism_project/requirements.txt

# Set HuggingFace token
export HF_TOKEN="your_huggingface_token"
```

### Running the Pipeline

1. **Data Registration**:
   ```bash
   cd tourism_project
   python data_registration.py
   ```

2. **Data Preparation**:
   ```bash
   python data_preparation.py
   ```

3. **Model Training**:
   ```bash
   python model_building.py
   ```

4. **Deployment**:
   ```bash
   cd deployment
   python deploy_to_hf.py
   ```

## Model Performance

The pipeline trains and compares multiple algorithms:
- **Decision Tree**: Baseline interpretable model
- **Random Forest**: Ensemble method for improved accuracy
- **Gradient Boosting**: Sequential boosting for complex patterns
- **XGBoost**: Optimized gradient boosting
- **AdaBoost**: Adaptive boosting for weak learners

Best model selection based on ROC-AUC score with comprehensive evaluation metrics.

## Web Application Features

The Streamlit app provides:
- Interactive customer input form
- Real-time prediction with confidence scores
- Probability breakdown visualization
- Customer profile summary
- User-friendly interface for business users

## CI/CD Automation

GitHub Actions workflow (`/.github/workflows/pipeline.yml`):
- **Trigger**: Push to main branch
- **Jobs**: 
  1. Data Registration
  2. Data Preparation
  3. Model Training with MLflow
  4. Automated Deployment to HuggingFace Spaces

## Key Insights

1. **Data Quality**: Handled missing values and data inconsistencies
2. **Feature Engineering**: Created meaningful categories for income and age
3. **Model Selection**: Ensemble methods outperformed single algorithms
4. **Automation**: End-to-end pipeline reduces manual intervention
5. **Scalability**: Containerized deployment ensures consistent environments

## Business Impact

- **Improved Targeting**: Predict high-likelihood customers before contact
- **Cost Efficiency**: Reduce marketing spend on low-probability prospects
- **Scalability**: Automated pipeline handles growing customer base
- **Adaptability**: Continuous model updates with new data
- **Decision Support**: Data-driven insights for marketing strategies

## Technical Stack

- **ML Framework**: Scikit-learn, XGBoost
- **Experiment Tracking**: MLflow
- **Data Platform**: HuggingFace Hub
- **Web Framework**: Streamlit
- **Containerization**: Docker
- **CI/CD**: GitHub Actions
- **Version Control**: Git/GitHub

## Usage Instructions

1. Clone this repository
2. Set up environment variables (HF_TOKEN)
3. Run the pipeline scripts in sequence
4. Access the deployed app on HuggingFace Spaces
5. Use GitHub Actions for automated deployments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit pull request
5. GitHub Actions will automatically test and deploy

---

**Project by**: Jyothi K
**MLOps Pipeline**: Tourism Package Prediction  
**Framework**: End-to-end automated ML workflow 
