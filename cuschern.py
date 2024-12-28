import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_data = pd.read_csv(r"D:\da\customer_churn_dataset-training-master.csv")
test_data = pd.read_csv(r"D:\da\customer_churn_dataset-testing-master.csv")

# Display the first few rows of the training dataframe
print(train_data.head())

# Display the first few rows of the testing dataframe
print(test_data.head())

# Display summary statistics for training data
print(train_data.describe())

# Display summary statistics for testing data
print(test_data.describe())

# Check for missing values in training data
print(train_data.isnull().sum())

# Check for missing values in testing data
print(test_data.isnull().sum())

# Fill missing values
train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

# Encode categorical variables
label_encoders = {}
for column in train_data.select_dtypes(include=['object']).columns:
    # Convert all values in the column to strings
    train_data[column] = train_data[column].astype(str)
    test_data[column] = test_data[column].astype(str)
    
    label_encoders[column] = LabelEncoder()
    train_data[column] = label_encoders[column].fit_transform(train_data[column])
    
    if column in test_data:
        test_data[column] = label_encoders[column].transform(test_data[column])

# Define features (X) and target (y) for training data
X_train = train_data.drop(columns=['Churn'])
y_train = train_data['Churn']

# Define features (X) and target (y) for testing data
X_test = test_data.drop(columns=['Churn'])
y_test = test_data['Churn']

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Print classification report
print(classification_report(y_test, y_pred))

# Calculate AUC-ROC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'AUC-ROC: {roc_auc}')

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='roc_auc')

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best AUC-ROC Score:", grid_search.best_score_)

# Train the best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predict on the test set using the best model
y_pred_best = best_model.predict(X_test)
y_pred_best_proba = best_model.predict_proba(X_test)[:, 1]

# Print classification report for the best model
print(classification_report(y_test, y_pred_best))

# Calculate AUC-ROC for the best model
roc_auc_best = roc_auc_score(y_test, y_pred_best_proba)
print(f'Best Model AUC-ROC: {roc_auc_best}')

# Plot ROC curve for the best model
fpr_best, tpr_best, thresholds_best = roc_curve(y_test, y_pred_best_proba)
plt.figure()
plt.plot(fpr_best, tpr_best, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_best)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
