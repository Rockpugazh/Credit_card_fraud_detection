
# Credit Card Fraud Detection System

![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-blueviolet)
![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)

A machine learning-based solution for detecting fraudulent credit card transactions in real-time.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview
This system uses advanced machine learning techniques to identify potentially fraudulent credit card transactions with high accuracy. It's designed to handle highly imbalanced datasets typical in fraud detection scenarios.

## Features
- 🚨 Real-time fraud detection
- ⚖️ Advanced handling of imbalanced data (SMOTE, ADASYN)
- 📊 Multiple ML algorithms (Random Forest, XGBoost, Isolation Forest)
- 📈 Comprehensive performance metrics
- 🔄 Feature engineering pipeline
- 🛡️ Model interpretability with SHAP values

## Dataset
The project uses the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle containing:
- 284,807 transactions (492 fraudulent)
- 30 numerical features (PCA-transformed for confidentiality)
- Transaction amounts from €0 to €25,691.16

## Installation

### Prerequisites
- Python 3.6+
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   credit-card-fraud-detection/
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Processed data files
├── models/                 # Saved model binaries
├── notebooks/              # Exploratory analysis
│   ├── EDA.ipynb
│   └── Model_Comparison.ipynb
├── src/
│   ├── preprocess.py       # Data cleaning
│   ├── train.py            # Model training
│   ├── evaluate.py         # Performance metrics
│   ├── predict.py          # Prediction interface
│   └── utils.py            # Helper functions
├── reports/                # Output visuals
├── requirements.txt        # Dependency list
├── LICENSE
└── README.md


#Model Performance
Model	Precision	Recall	F1-Score	ROC-AUC
Random Forest	0.92	0.85	0.88	0.98
XGBoost	0.94	0.83	0.88	0.97
Logistic Reg.	0.76	0.65	0.70	0.92
