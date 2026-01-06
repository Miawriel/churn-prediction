"""
Data preprocessing for bank churn prediction project.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(csv_path):
    """
    Loads the dataset, preprocesses features, and returns
    scaled train-test splits ready for model training.
    """

    # Load dataset
    df = pd.read_csv(csv_path)

    # Drop unnecessary identifier columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Encode categorical variables
    df = pd.get_dummies(
        df, columns=['Geography', 'Gender'], drop_first=True
    )

    # Split features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

