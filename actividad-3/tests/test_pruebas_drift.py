"""Tests de las pruebas estadísticas de drift por variable (domain/pruebas_drift.py)."""
import numpy as np
import pandas as pd
import pytest

from domain.modelo import VARIABLES
from domain.monitoreo import construir_lote_con_drift, dividir
from domain.pruebas_drift import (
    PSI_IMPORTANTE,
    PSI_MODERADO,
    drift_por_variable,
    evaluar_variable,
    fraccion_con_drift,
    prueba_chi2,
    prueba_ks,
    psi_continua,
    psi_discreta,
    severidad_psi,
)

DIVISION = {"test_size": 0.2, "random_state": 42}


@pytest.fixture
def rng():
    return np.random.default_rng(123)


# --- PSI ---


def test_psi_continua_es_casi_cero_con_la_misma_distribucion(rng):
    referencia = rng.normal(0, 1, 20_000)
    lote = rng.normal(0, 1, 2_000)
    assert psi_continua(referencia, lote) < PSI_MODERADO


def test_psi_continua_crece_con_el_desplazamiento(rng):
    referencia = rng.normal(0, 1, 20_000)
    psi_chico = psi_continua(referencia, rng.normal(0.3, 1, 2_000))
    psi_grande = psi_continua(referencia, rng.normal(2.0, 1, 2_000))
    assert psi_chico < psi_grande
    assert psi_grande > PSI_IMPORTANTE


def test_psi_continua_de_variable_constante_es_cero():
    assert psi_continua(np.zeros(100), np.zeros(50)) == 0.0


def test_psi_discreta_detecta_cambio_de_frecuencias():
    referencia = pd.Series([0] * 800 + [1] * 200)
    igual = pd.Series([0] * 80 + [1] * 20)
    distinta = pd.Series([0] * 30 + [1] * 70)
    assert psi_discreta(referencia, igual) < PSI_MODERADO
    assert psi_discreta(referencia, distinta) > PSI_IMPORTANTE


def test_psi_discreta_tolera_categoria_que_solo_existe_en_un_lado():
    referencia = pd.Series([1] * 100)
    lote = pd.Series([1] * 50 + [2] * 50)  # aparece la categoría 2, que no existía
    assert np.isfinite(psi_discreta(referencia, lote))
    assert psi_discreta(referencia, lote) > PSI_IMPORTANTE


@pytest.mark.parametrize("psi, esperado", [(0.02, "estable"), (0.15, "moderado"), (0.40, "importante")])
def test_severidad_psi(psi, esperado):
    assert severidad_psi(psi) == esperado


# --- KS y chi-cuadrado ---


def test_ks_detecta_desplazamiento_y_no_alerta_sin_el(rng):
    referencia = rng.normal(0, 1, 5_000)
    _, p_igual = prueba_ks(referencia, rng.normal(0, 1, 500))
    _, p_distinta = prueba_ks(referencia, rng.normal(1, 1, 500))
    assert p_igual > 0.01
    assert p_distinta < 1e-6


def test_chi2_detecta_cambio_de_frecuencias():
    referencia = pd.Series([0] * 800 + [1] * 200)
    _, p_igual = prueba_chi2(referencia, pd.Series([0] * 80 + [1] * 20))
    _, p_distinta = prueba_chi2(referencia, pd.Series([0] * 50 + [1] * 50))
    assert p_igual > 0.5
    assert p_distinta < 0.001


def test_chi2_agrupa_categorias_raras_sin_romperse():
    # La categoría 8 es rarísima (como PAY_0 = 8 en el dataset real): se agrupa en 'otras'
    referencia = pd.Series([0] * 9_000 + [1] * 990 + [8] * 10)
    lote = pd.Series([0] * 450 + [1] * 50)
    estadistico, p_valor = prueba_chi2(referencia, lote)
    assert np.isfinite(estadistico) and 0.0 <= p_valor <= 1.0


def test_chi2_con_una_sola_categoria_no_hay_nada_que_comparar():
    assert prueba_chi2(pd.Series([3] * 100), pd.Series([3] * 50)) == (0.0, 1.0)


# --- evaluar_variable: significancia + magnitud ---


def test_efecto_diminuto_no_cuenta_como_drift_aunque_sea_significativo(rng):
    """Con muestras grandes, un cambio mínimo da un p-valor chiquísimo pero no le importa a nadie:
    por eso se exige también un PSI mínimo (tamaño de efecto)."""
    referencia = pd.Series(rng.normal(0, 1, 50_000))
    lote = pd.Series(rng.normal(0.08, 1, 50_000))
    resultado = evaluar_variable("AGE", referencia, lote, alfa=0.05, psi_minimo=PSI_MODERADO)
    assert resultado["p_valor"] < 0.05  # estadísticamente significativo...
    assert resultado["psi"] < PSI_MODERADO  # ...pero irrelevante en la práctica
    assert resultado["drift"] is False


def test_cambio_grande_si_cuenta_como_drift(rng):
    referencia = pd.Series(rng.normal(0, 1, 5_000))
    lote = pd.Series(rng.normal(1.5, 1, 500))
    resultado = evaluar_variable("AGE", referencia, lote, alfa=0.05, psi_minimo=PSI_MODERADO)
    assert resultado["drift"] is True
    assert resultado["prueba"] == "ks" and resultado["tipo"] == "continua"


def test_variable_discreta_usa_chi2():
    referencia = pd.Series([0] * 800 + [1] * 200)
    lote = pd.Series([0] * 50 + [1] * 50)
    resultado = evaluar_variable("SEX", referencia, lote, alfa=0.05, psi_minimo=PSI_MODERADO)
    assert resultado["prueba"] == "chi2" and resultado["tipo"] == "discreta"
    assert resultado["drift"] is True


# --- Integración de la lógica de monitoreo (sin servicios) ---


def _tabla_para_nivel(datos_crudos, nivel):
    from domain.ingenieria import agregar_variables_derivadas

    entrenamiento, _ = dividir(datos_crudos, DIVISION)
    referencia = agregar_variables_derivadas(entrenamiento)[VARIABLES]
    lote = construir_lote_con_drift(datos_crudos, DIVISION, tamano_lote=500, nivel_drift=nivel)
    return drift_por_variable(referencia, lote, alfa=0.05, psi_minimo=PSI_MODERADO)


def test_sin_drift_emulado_casi_ninguna_variable_se_marca(datos_crudos):
    tabla = _tabla_para_nivel(datos_crudos, 0.0)
    assert fraccion_con_drift(tabla) < 0.30  # por debajo del umbral de alerta de config.yaml


def test_con_drift_alto_se_supera_el_umbral_de_alerta(datos_crudos):
    tabla = _tabla_para_nivel(datos_crudos, 0.6)
    assert fraccion_con_drift(tabla) > 0.30


def test_mas_drift_emulado_marca_mas_variables(datos_crudos):
    f_bajo = fraccion_con_drift(_tabla_para_nivel(datos_crudos, 0.3))
    f_alto = fraccion_con_drift(_tabla_para_nivel(datos_crudos, 0.6))
    assert f_alto > f_bajo


def test_la_tabla_viene_ordenada_por_psi_y_cubre_todas_las_variables(datos_crudos):
    tabla = _tabla_para_nivel(datos_crudos, 0.6)
    assert list(tabla["psi"]) == sorted(tabla["psi"], reverse=True)
    assert set(tabla["variable"]) == set(VARIABLES)
