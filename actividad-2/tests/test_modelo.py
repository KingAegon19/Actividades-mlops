import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from domain.modelo import calcular_metricas, construir_modelo, debe_promover_a_champion, resolver_features


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


def test_debe_promover_sin_champion_previo_cumple_ambos_umbrales():
    metricas = {"auc": 0.75, "recall": 0.55}
    assert debe_promover_a_champion(metricas, umbral_auc=0.70, umbral_recall=0.50, metricas_champion_actual=None)


def test_debe_promover_rechaza_por_auc_bajo():
    metricas = {"auc": 0.65, "recall": 0.90}
    assert not debe_promover_a_champion(metricas, umbral_auc=0.70, umbral_recall=0.50, metricas_champion_actual=None)


def test_debe_promover_rechaza_por_recall_bajo_aunque_auc_sea_alto():
    metricas = {"auc": 0.95, "recall": 0.20}
    assert not debe_promover_a_champion(metricas, umbral_auc=0.70, umbral_recall=0.50, metricas_champion_actual=None)


def test_debe_promover_rechaza_si_no_supera_al_champion_actual():
    metricas = {"auc": 0.72, "recall": 0.60}
    champion_actual = {"auc": 0.80}
    assert not debe_promover_a_champion(metricas, umbral_auc=0.70, umbral_recall=0.50, metricas_champion_actual=champion_actual)


def test_debe_promover_acepta_si_supera_al_champion_actual():
    metricas = {"auc": 0.85, "recall": 0.60}
    champion_actual = {"auc": 0.80}
    assert debe_promover_a_champion(metricas, umbral_auc=0.70, umbral_recall=0.50, metricas_champion_actual=champion_actual)
