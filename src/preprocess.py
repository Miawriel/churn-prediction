"""
Data preprocessing for bank churn prediction project.
"""
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

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)

    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

    X = df.drop('Exited', axis=1)
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns



