import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from domain.modelo import (
    calcular_metricas,
    construir_modelo,
    debe_promover_a_champion,
    elegir_mejor_candidato,
    evaluar_con_kfold,
    resolver_features,
)


def test_resolver_features_all():
    numericas, categoricas = resolver_features("ALL")
    assert "LIMIT_BAL" in numericas
    assert "SEX" in categoricas
    # 23 columnas originales + 8 derivadas de domain/ingenieria.py (extra opcional)
    assert len(numericas) + len(categoricas) == 31


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


def test_elegir_mejor_candidato_descarta_el_de_mejor_auc_si_no_pasa_recall():
    # El de mejor AUC (0.80) no pasa el umbral de recall; el segundo (0.78) sí -- debe ganar
    # el segundo, no quedarse sin ganador como pasaría si solo se mirara el mejor AUC de la tanda.
    resultados = [
        ("uri-alto-auc", {"auc": 0.80, "recall": 0.30}),
        ("uri-balanceado", {"auc": 0.78, "recall": 0.60}),
    ]
    ganador = elegir_mejor_candidato(resultados, umbral_auc=0.70, umbral_recall=0.50, metricas_champion_actual=None)
    assert ganador[0] == "uri-balanceado"


def test_elegir_mejor_candidato_ninguno_califica_devuelve_none():
    resultados = [
        ("uri-1", {"auc": 0.60, "recall": 0.30}),
        ("uri-2", {"auc": 0.65, "recall": 0.20}),
    ]
    assert elegir_mejor_candidato(resultados, umbral_auc=0.70, umbral_recall=0.50, metricas_champion_actual=None) is None


def test_elegir_mejor_candidato_entre_varios_validos_gana_el_de_mejor_auc():
    resultados = [
        ("uri-1", {"auc": 0.75, "recall": 0.55}),
        ("uri-2", {"auc": 0.80, "recall": 0.60}),
    ]
    ganador = elegir_mejor_candidato(resultados, umbral_auc=0.70, umbral_recall=0.50, metricas_champion_actual=None)
    assert ganador[0] == "uri-2"


def test_evaluar_con_kfold_devuelve_promedio_y_std_de_cada_metrica():
    # Datos separables con una sola variable: un modelo decente debería acercarse a AUC=1
    # en casi todos los folds, así que el promedio también debería ser alto.
    X = pd.DataFrame({"LIMIT_BAL": list(range(0, 100))})
    y = pd.Series([0] * 50 + [1] * 50)
    pipeline = construir_modelo("logistic_regression", {"max_iter": 500}, ["LIMIT_BAL"], [])

    resultado = evaluar_con_kfold(pipeline, X, y, umbral=0.5, n_splits=5, random_state=42)

    assert set(resultado) == {"auc_kfold_promedio", "auc_kfold_std", "recall_kfold_promedio", "recall_kfold_std"}
    assert resultado["auc_kfold_promedio"] > 0.9
    assert 0.0 <= resultado["recall_kfold_promedio"] <= 1.0


def test_evaluar_con_kfold_no_entrena_el_pipeline_original():
    # El pipeline que se pasa debe seguir sin entrenar después de llamar a la función --
    # el que se registra en MLflow se entrena aparte, con el fit explícito en el flow.
    X = pd.DataFrame({"LIMIT_BAL": list(range(0, 100))})
    y = pd.Series([0] * 50 + [1] * 50)
    pipeline = construir_modelo("logistic_regression", {"max_iter": 500}, ["LIMIT_BAL"], [])

    evaluar_con_kfold(pipeline, X, y, umbral=0.5, n_splits=5, random_state=42)

    with pytest.raises(Exception):
        pipeline.predict_proba(X)
