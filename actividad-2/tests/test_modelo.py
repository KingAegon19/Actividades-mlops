import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from domain.modelo import calcular_metricas, construir_modelo, resolver_features


def test_resolver_features_all():
    numericas, categoricas = resolver_features("ALL")
    assert "LIMIT_BAL" in numericas
    assert "SEX" in categoricas
    assert len(numericas) + len(categoricas) == 23


def test_resolver_features_custom():
    numericas, categoricas = resolver_features(["LIMIT_BAL", "PAY_0", "SEX"])
    assert numericas == ["LIMIT_BAL", "PAY_0"]
    assert categoricas == ["SEX"]


def test_resolver_features_variable_desconocida():
    with pytest.raises(ValueError):
        resolver_features(["COLUMNA_QUE_NO_EXISTE"])


def test_construir_modelo_logistic_regression():
    modelo = construir_modelo("logistic_regression", {"max_iter": 500}, ["LIMIT_BAL"], ["SEX"])
    assert isinstance(modelo.named_steps["modelo"], LogisticRegression)


def test_construir_modelo_random_forest():
    modelo = construir_modelo("random_forest", {"random_state": 42}, ["LIMIT_BAL"], ["SEX"])
    assert isinstance(modelo.named_steps["modelo"], RandomForestClassifier)


def test_construir_modelo_desconocido():
    with pytest.raises(ValueError):
        construir_modelo("modelo_inexistente", {}, ["LIMIT_BAL"], [])


def test_calcular_metricas_caso_perfecto():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.8])
    metricas = calcular_metricas(y_true, proba, umbral=0.5)
    assert metricas["auc"] == 1.0
    assert metricas["accuracy"] == 1.0
    assert set(metricas) == {"auc", "accuracy", "precision", "recall", "f1"}


def test_calcular_metricas_umbral_mas_bajo_sube_recall():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.35, 0.4, 0.8])
    metricas_umbral_alto = calcular_metricas(y_true, proba, umbral=0.5)
    metricas_umbral_bajo = calcular_metricas(y_true, proba, umbral=0.3)
    assert metricas_umbral_bajo["recall"] >= metricas_umbral_alto["recall"]
