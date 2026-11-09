import math

import pandas as pd
import pytest

from domain.ingenieria import agregar_variables_derivadas


def _fila_base(**overrides):
    fila = {
        "LIMIT_BAL": 100000.0,
        "BILL_AMT1": 20000.0, "BILL_AMT2": 20000.0, "BILL_AMT3": 20000.0,
        "BILL_AMT4": 20000.0, "BILL_AMT5": 20000.0, "BILL_AMT6": 20000.0,
        "PAY_AMT1": 10000.0, "PAY_AMT2": 10000.0, "PAY_AMT3": 10000.0,
        "PAY_AMT4": 10000.0, "PAY_AMT5": 10000.0, "PAY_AMT6": 10000.0,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": 0, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
    }
    fila.update(overrides)
    return fila


def test_agrega_las_ocho_columnas_derivadas():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base()]))
    columnas_esperadas = [
        "meses_con_atraso",
        "atraso_promedio",
        "utilizacion_promedio",
        "interaccion_atraso_x_utilizacion",
        "interaccion_atraso_x_limite",
        "atraso_promedio_cuadrado",
        "log_limit_bal",
        "razon_pago_reciente",
    ]
    for columna in columnas_esperadas:
        assert columna in df.columns


def test_utilizacion_promedio_es_bill_sobre_limite():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base()]))
    # BILL_AMT promedio = 20000, LIMIT_BAL = 100000 -> 0.2
    assert df["utilizacion_promedio"].iloc[0] == pytest.approx(0.2)


def test_meses_con_atraso_cuenta_estados_positivos():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base()]))
    # PAY_0=2 y PAY_2=2 son los unicos > 0 de los 6 meses
    assert df["meses_con_atraso"].iloc[0] == 2


def test_no_falla_con_limite_de_credito_cero():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base(LIMIT_BAL=0.0)]))
    assert df["utilizacion_promedio"].iloc[0] == 0.0


def test_no_modifica_el_dataframe_original():
    original = pd.DataFrame([_fila_base()])
    agregar_variables_derivadas(original)
    assert "utilizacion_promedio" not in original.columns


def test_interaccion_atraso_x_utilizacion_es_el_producto_de_las_dos():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base()]))
    # meses_con_atraso=2, utilizacion_promedio=0.2 -> 0.4
    assert df["interaccion_atraso_x_utilizacion"].iloc[0] == pytest.approx(0.4)


def test_interaccion_atraso_x_limite_es_el_producto_de_las_dos():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base()]))
    # meses_con_atraso=2, LIMIT_BAL=100000 -> 200000
    assert df["interaccion_atraso_x_limite"].iloc[0] == pytest.approx(200000)


def test_atraso_promedio_cuadrado_es_el_cuadrado_del_atraso_promedio():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base()]))
    atraso_promedio = df["atraso_promedio"].iloc[0]
    assert df["atraso_promedio_cuadrado"].iloc[0] == pytest.approx(atraso_promedio**2)


def test_log_limit_bal_es_log1p_del_limite():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base()]))
    assert df["log_limit_bal"].iloc[0] == pytest.approx(math.log1p(100000.0))


def test_razon_pago_reciente_es_pay_amt1_sobre_bill_amt1():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base()]))
    # PAY_AMT1=10000, BILL_AMT1=20000 -> 0.5
    assert df["razon_pago_reciente"].iloc[0] == pytest.approx(0.5)


def test_razon_pago_reciente_no_falla_si_bill_amt1_es_cero():
    df = agregar_variables_derivadas(pd.DataFrame([_fila_base(BILL_AMT1=0.0)]))
    assert df["razon_pago_reciente"].iloc[0] == 0.0
