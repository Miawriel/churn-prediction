# Data Loading
import pandas as pd

df = pd.read_csv('Churn_Modelling.csv')
print(df.head())
print(df.info())

# Target variable exploration
print(df['Exited'].value_counts())
# This shows how many customers churned (1) vs. stayed (0)

# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming the DataFrame is called 'df'

# 1. DROP UNNECESSARY COLUMNS
print("Step 1: Dropping identifier columns...")
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# 2. ENCODE CATEGORICAL VARIABLES (One-Hot Encoding)
# 'Geography' and 'Gender' are categorical variables and must be converted to numeric
# drop_first=True is used to avoid multicollinearity (dummy variable trap)
print("Step 2: Encoding categorical variables...")
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
print("Columns after encoding:", df.columns.tolist())
print("-" * 30)

# 3. SPLIT FEATURES (X) AND TARGET (y)
# X contains all features except 'Exited'
X = df.drop('Exited', axis=1)

# y is the target variable
y = df['Exited']
print("Step 3: Splitting features and target. X has", X.shape[1], "features.")

# 4. TRAIN-TEST SPLIT
# Data is split before scaling to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Step 4: Data split completed. Train:", X_train.shape, "Test:", X_test.shape)

# 5. FEATURE SCALING
# Scaling prevents features like 'EstimatedSalary' from dominating others
scaler = StandardScaler()

# 5a. Fit scaler using training data only
X_train_scaled = scaler.fit_transform(X_train)

# 5b. Apply the same transformation to test data
X_test_scaled = scaler.transform(X_test)
print("Step 5: Feature scaling completed.")
print("-" * 30)

print("✅ PREPROCESSING COMPLETED")

# Training the Random Forest model
from sklearn.ensemble import RandomForestClassifier

# 6. MODEL TRAINING
# random_state is used for reproducibility
rf_model = RandomForestClassifier(random_state=42)

print("Step 6: Training Random Forest model...")
rf_model.fit(X_train_scaled, y_train)

print("✅ Model training completed.")

# Making predictions with Random Forest
# 7. GENERATE PREDICTIONS ON TEST SET
y_pred_rf = rf_model.predict(X_test_scaled)
print("Predictions generated.")

# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print("\n--- Model Evaluation Results ---")

# a) Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
# [[True Negatives, False Positives],
#  [False Negatives, True Positives]]

# b) Classification Report (Accuracy, Precision, Recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
# Pay special attention to Recall for class '1' (churned customers)

# c) ROC-AUC Score
auc_score = roc_auc_score(
    y_test, rf_model.predict_proba(X_test_scaled)[:, 1]
)
print(f"\nAUC (Area Under the Curve): {auc_score:.4f}")

# Risk Factors Analysis
# Extract feature importance from Random Forest
feature_importances = pd.Series(
    rf_model.feature_importances_, index=X.columns
)

# Display top 10 most important features
top_10_features = feature_importances.nlargest(10)
print("\n--- Top 10 Risk Factors (Feature Importance) ---")
print(top_10_features)

# Optional: Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
top_10_features.plot(kind='barh')
plt.title('Feature Importance for Churn Prediction')
plt.show()

