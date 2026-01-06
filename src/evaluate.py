"""
Model evaluation and comparison for bank churn prediction project.
"""

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from train import train_models


def evaluate_models(csv_path):
    """
    Evaluates multiple trained models and compares performance.
    """

    models, X_test, y_test, feature_names = train_models(csv_path)

    results = {}

    for name, model in models.items():
        print(f"\n=== {name.upper()} RESULTS ===")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        auc = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC: {auc:.4f}")

        results[name] = auc

    return results


if __name__ == "__main__":
    evaluate_models("Churn_Modelling.csv")

