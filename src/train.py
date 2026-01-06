"""
Model training for bank churn prediction project.
"""

from sklearn.ensemble import RandomForestClassifier
from preprocess import preprocess_data


def train_model(csv_path):
    """
    Trains a Random Forest model using preprocessed data.
    """

    X_train, X_test, y_train, y_test, feature_names = preprocess_data(csv_path)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, feature_names


if __name__ == "__main__":
    model, _, _, _ = train_model("Churn_Modelling.csv")
    print("Model training completed successfully.")

