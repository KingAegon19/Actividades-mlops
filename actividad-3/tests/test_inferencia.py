"""Tests de la lógica de inferencia (tarea 3 de docs/actividad-3.md): armado del lote con drift y
puntuación. Son las funciones puras que `inferencia_flow.py` orquesta -- se prueban sin Prefect
ni MLflow."""
import numpy as np
import pytest

from domain.ingenieria import VARIABLES_DERIVADAS
from domain.modelo import OBJETIVO, VARIABLES, construir_modelo, puntuar_lote, resolver_features
from domain.monitoreo import construir_lote_con_drift, dividir

DIVISION = {"test_size": 0.2, "random_state": 42}


def _lote(datos_crudos, nivel=0.0, tamano=200):
    return construir_lote_con_drift(datos_crudos, DIVISION, tamano_lote=tamano, nivel_drift=nivel)


# --- construir_lote_con_drift ---


def test_el_lote_tiene_el_tamano_pedido(datos_crudos):
    assert len(_lote(datos_crudos, tamano=150)) == 150


def test_el_lote_viene_solo_de_la_particion_de_prueba(datos_crudos):
    """Garantía de monitoreo: el lote nunca contiene filas con las que se entrenó el modelo."""
    entrenamiento, prueba = dividir(datos_crudos, DIVISION)
    lote = _lote(datos_crudos, nivel=0.0)
    assert set(lote.index) <= set(prueba.index)
    assert not set(lote.index) & set(entrenamiento.index)


def test_el_lote_incluye_variables_derivadas_y_objetivo(datos_crudos):
    lote = _lote(datos_crudos)
    assert set(VARIABLES_DERIVADAS) <= set(lote.columns)
    assert set(VARIABLES) <= set(lote.columns)
    assert OBJETIVO in lote.columns


def test_el_lote_es_reproducible(datos_crudos):
    assert _lote(datos_crudos, nivel=0.6).equals(_lote(datos_crudos, nivel=0.6))


def test_sin_drift_el_lote_es_igual_a_la_muestra_original(datos_crudos):
    _, prueba = dividir(datos_crudos, DIVISION)
    lote = _lote(datos_crudos, nivel=0.0)
    original = prueba.loc[lote.index]
    for columna in original.columns:
        assert np.allclose(lote[columna], original[columna])


def test_con_drift_cambian_las_columnas_derivadas_tambien(datos_crudos):
    """Las derivadas se calculan DESPUÉS del drift, así que deben reflejarlo."""
    sin_drift = _lote(datos_crudos, nivel=0.0)
    con_drift = _lote(datos_crudos, nivel=0.6)
    cambiadas = [c for c in VARIABLES_DERIVADAS if not np.allclose(sin_drift[c], con_drift[c])]
    assert cambiadas, "ninguna variable derivada reflejó el drift"


def test_con_drift_sube_la_tasa_de_default(datos_crudos):
    assert _lote(datos_crudos, nivel=0.6)[OBJETIVO].mean() > _lote(datos_crudos, nivel=0.0)[OBJETIVO].mean()


def test_nivel_de_drift_invalido_se_rechaza(datos_crudos):
    with pytest.raises(ValueError):
        _lote(datos_crudos, nivel=1.5)


# --- puntuar_lote ---


@pytest.fixture
def modelo_entrenado(datos_crudos):
    lote = _lote(datos_crudos, nivel=0.0, tamano=400)
    numericas, categoricas = resolver_features("ALL")
    modelo = construir_modelo("logistic_regression", {"max_iter": 500}, numericas, categoricas)
    return modelo.fit(lote[numericas + categoricas], lote[OBJETIVO])


def test_puntuar_lote_agrega_probabilidad_y_prediccion(modelo_entrenado, datos_crudos):
    lote = _lote(datos_crudos)
    resultado = puntuar_lote(modelo_entrenado, lote, umbral=0.5)
    assert len(resultado) == len(lote)
    assert resultado["probabilidad_default"].between(0, 1).all()
    assert set(resultado["prediccion"]) <= {0, 1}


def test_la_prediccion_respeta_el_umbral(modelo_entrenado, datos_crudos):
    lote = _lote(datos_crudos)
    estricto = puntuar_lote(modelo_entrenado, lote, umbral=0.9)["prediccion"].sum()
    laxo = puntuar_lote(modelo_entrenado, lote, umbral=0.1)["prediccion"].sum()
    assert laxo > estricto


def test_puntuar_lote_no_modifica_el_lote_original(modelo_entrenado, datos_crudos):
    lote = _lote(datos_crudos)
    columnas = list(lote.columns)
    puntuar_lote(modelo_entrenado, lote, umbral=0.5)
    assert list(lote.columns) == columnas


def test_puntuar_lote_ignora_columnas_de_mas(modelo_entrenado, datos_crudos):
    """El lote trae el objetivo y otras columnas que el modelo no usa: no deben romper la predicción."""
    lote = _lote(datos_crudos)
    lote["columna_ajena"] = 1
    resultado = puntuar_lote(modelo_entrenado, lote, umbral=0.5)
    assert "columna_ajena" in resultado.columns  # se conserva, solo no se usa para predecir


def test_puntuar_lote_falla_si_falta_una_variable_del_modelo(modelo_entrenado, datos_crudos):
    lote = _lote(datos_crudos).drop(columns=["AGE"])
    with pytest.raises(KeyError):
        puntuar_lote(modelo_entrenado, lote, umbral=0.5)
