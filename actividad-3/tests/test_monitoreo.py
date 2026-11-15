"""Tests de la lógica de drift (domain/monitoreo.py). No necesitan MLflow ni Prefect."""
import numpy as np
import pandas as pd
import pytest

from domain.modelo import NUMERICAS_CRUDAS, OBJETIVO
from domain.monitoreo import (
    SUBIDA_TASA_MAXIMA,
    aplicar_drift,
    drift_de_target,
    fraccion_desde_evidently,
    validar_lote,
)


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({columna: rng.normal(0, 1, 500) for columna in NUMERICAS_CRUDAS})
    df[OBJETIVO] = rng.integers(0, 2, 500)
    return df


def test_drift_cero_no_cambia_los_datos(datos):
    assert aplicar_drift(datos, 0.0).equals(datos)


def test_drift_rechaza_nivel_fuera_de_rango(datos):
    with pytest.raises(ValueError):
        aplicar_drift(datos, 1.5)


def test_drift_sube_la_tasa_del_objetivo_en_proporcion_al_nivel(datos):
    tasa = datos[OBJETIVO].mean()
    lote = aplicar_drift(datos, 0.6)
    assert lote[OBJETIVO].mean() == pytest.approx(tasa + 0.6 * SUBIDA_TASA_MAXIMA, abs=0.01)


def test_drift_cambia_la_fraccion_indicada_de_variables(datos):
    lote = aplicar_drift(datos, 0.6)
    cambiadas = [c for c in NUMERICAS_CRUDAS if not np.allclose(lote[c], datos[c])]
    assert len(cambiadas) == round(0.6 * len(NUMERICAS_CRUDAS))


def test_drift_mas_alto_cambia_mas_variables(datos):
    lote_bajo = aplicar_drift(datos, 0.3)
    lote_alto = aplicar_drift(datos, 0.6)
    cambiadas_bajo = sum(1 for c in NUMERICAS_CRUDAS if not np.allclose(lote_bajo[c], datos[c]))
    cambiadas_alto = sum(1 for c in NUMERICAS_CRUDAS if not np.allclose(lote_alto[c], datos[c]))
    assert cambiadas_alto > cambiadas_bajo


def test_fraccion_desde_evidently():
    resultado = {
        "metrics": [
            {"metric_name": "ValueDrift(column=AGE)", "value": 0.5},
            {"metric_name": "DriftedColumnsCount(drift_share=0.5)", "value": {"count": 7, "share": 0.5}},
        ]
    }
    assert fraccion_desde_evidently(resultado) == 0.5


def test_fraccion_desde_evidently_exige_la_metrica():
    with pytest.raises(ValueError):
        fraccion_desde_evidently({"metrics": [{"metric_name": "ValueDrift(column=AGE)", "value": 0.5}]})


def test_validar_lote_exige_objetivo(datos):
    with pytest.raises(ValueError, match=OBJETIVO.replace(".", r"\.")):
        validar_lote(datos.drop(columns=[OBJETIVO]))


def test_validar_lote_exige_objetivo_sin_nulos(datos):
    datos.loc[0, OBJETIVO] = None
    with pytest.raises(ValueError, match=OBJETIVO.replace(".", r"\.")):
        validar_lote(datos)


def test_drift_de_target_detecta_cambio_de_tasa():
    referencia = pd.Series([0] * 800 + [1] * 200)
    lote = pd.Series([0] * 400 + [1] * 600)
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is True


def test_drift_de_target_no_alerta_con_la_misma_tasa():
    referencia = pd.Series([0] * 800 + [1] * 200)
    lote = pd.Series([0] * 80 + [1] * 20)
    assert drift_de_target(referencia, lote, alfa=0.05)["drift"] is False
