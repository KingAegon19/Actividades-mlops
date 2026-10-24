import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def partir_datos(df: pd.DataFrame, objetivo: str, test_size: float, random_state: int):
    """Partición estratificada por el target: el dataset está desbalanceado (~22% positivos)."""
    X = df.drop(columns=[objetivo])
    y = df[objetivo]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def calcular_metricas_binarias(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
