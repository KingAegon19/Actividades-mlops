import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.modelo import obtener_modelo


def test_obtener_modelo_logistic_regression():
    modelo = obtener_modelo("logistic_regression", random_state=42)
    assert isinstance(modelo, LogisticRegression)


def test_obtener_modelo_random_forest():
    modelo = obtener_modelo("random_forest", random_state=42)
    assert isinstance(modelo, RandomForestClassifier)


def test_obtener_modelo_desconocido_lanza_error():
    with pytest.raises(ValueError):
        obtener_modelo("modelo_inexistente", random_state=42)
