# Importing Dependencies
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

# Load Dataset
credit_card_data = pd.read_csv('/content/Credit_Card_Fraud_Detection.csv')

# Display basic info
print(credit_card_data.head())
print(credit_card_data.info())

# Check missing values
print(credit_card_data.isnull().sum())

# Check class distribution
print(credit_card_data['class'].value_counts())

# Separate Legit and Fraud transactions
legit = credit_card_data[credit_card_data['class'] == 0]
fraud = credit_card_data[credit_card_data['class'] == 1]

print("Legit transactions:", legit.shape)
print("Fraud transactions:", fraud.shape)

# Under Sampling (to balance the dataset)
legit_sample = legit.sample(n=len(fraud), random_state=42)

# Combine the two datasets
new_dataset = pd.concat([legit_sample, fraud], axis=0)

print(new_dataset['class'].value_counts())

# Drop unnecessary columns
new_dataset = new_dataset.drop(columns=['Unnamed: 0', 'Customer_ID'])

# Splitting Features and Target
X = new_dataset.drop(columns='class', axis=1)
Y = new_dataset['class']

# Split the data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=42
)

# -------------------------
# Model Training
# -------------------------

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, Y_train)

# -------------------------
# Training Performance
# -------------------------

train_pred = model.predict(X_train)
train_probs = model.predict_proba(X_train)[:, 1]

print("=== Training Set Performance ===")
print("Accuracy:", accuracy_score(Y_train, train_pred))
print("Precision:", precision_score(Y_train, train_pred))
print("Recall:", recall_score(Y_train, train_pred))
print("F1 Score:", f1_score(Y_train, train_pred))
print("ROC AUC:", roc_auc_score(Y_train, train_probs))

print("\nClassification Report:")
print(classification_report(Y_train, train_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_train, train_pred))

# -------------------------
# Test Performance
# -------------------------

test_pred = model.predict(X_test)
test_probs = model.predict_proba(X_test)[:, 1]

print("\n=== Test Set Performance ===")
print("Accuracy:", accuracy_score(Y_test, test_pred))
print("Precision:", precision_score(Y_test, test_pred))
print("Recall:", recall_score(Y_test, test_pred))
print("F1 Score:", f1_score(Y_test, test_pred))
print("ROC AUC:", roc_auc_score(Y_test, test_probs))

print("\nClassification Report:")
print(classification_report(Y_test, test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, test_pred))
