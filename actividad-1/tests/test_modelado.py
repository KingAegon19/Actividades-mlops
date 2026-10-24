import pandas as pd

from src.modelado import calcular_metricas_binarias, partir_datos


def test_partir_datos_mantiene_proporcion_estratificada():
    df = pd.DataFrame(
        {
            "feature": range(20),
            "default.payment.next.month": [0] * 15 + [1] * 5,
        }
    )
    X_train, X_test, y_train, y_test = partir_datos(
        df, "default.payment.next.month", test_size=0.5, random_state=42
    )

    assert len(X_train) == len(X_test) == 10
    assert y_train.sum() == 2 or y_train.sum() == 3  # proporción ~25% preservada por el estrato
    assert y_train.sum() + y_test.sum() == 5


def test_calcular_metricas_binarias_caso_perfecto():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.9, 0.8]

    metricas = calcular_metricas_binarias(y_true, y_pred, y_proba)

    assert metricas["accuracy"] == 1.0
    assert metricas["auc"] == 1.0
    assert set(metricas) == {"accuracy", "auc", "precision", "recall", "f1"}
