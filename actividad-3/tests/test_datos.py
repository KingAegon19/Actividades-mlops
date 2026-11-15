from pathlib import Path

import pandas as pd
import pytest

from src.datos import agrupar_categorias_sin_documentar, cargar_datos, validar_calidad

FIXTURE = Path(__file__).parent / "fixtures" / "credit_sample.csv"
OBJETIVO = "default.payment.next.month"


def test_cargar_datos_descarta_id():
    df = cargar_datos(FIXTURE, OBJETIVO)
    assert "ID" not in df.columns
    assert len(df) == 10


def test_cargar_datos_agrupa_categorias_sin_documentar():
    df = cargar_datos(FIXTURE, OBJETIVO)
    assert set(df["EDUCATION"]) <= {1, 2, 3, 4}
    assert set(df["MARRIAGE"]) <= {1, 2, 3}


def test_cargar_datos_falla_si_falta_objetivo():
    with pytest.raises(ValueError):
        cargar_datos(FIXTURE, "columna_que_no_existe")


def test_agrupar_categorias_sin_documentar():
    df = pd.DataFrame({"EDUCATION": [0, 5, 6, 1, 2], "MARRIAGE": [0, 1, 2, 3, 0]})
    df_agrupado = agrupar_categorias_sin_documentar(df)

    assert set(df_agrupado["EDUCATION"]) <= {1, 2, 4}
    assert set(df_agrupado["MARRIAGE"]) <= {1, 2, 3}


# --- Calidad de datos: esquema, nulos y rangos (tarea 3 de docs/actividad-3.md) ---


@pytest.fixture
def df_valido():
    return cargar_datos(FIXTURE, OBJETIVO)


def test_dataset_valido_no_tiene_problemas(df_valido):
    assert validar_calidad(df_valido, OBJETIVO) == []


def test_cargar_datos_falla_si_falta_una_columna_del_esquema(tmp_path, df_valido):
    ruta = tmp_path / "sin_age.csv"
    pd.read_csv(FIXTURE).drop(columns=["AGE"]).to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="AGE"):
        cargar_datos(ruta, OBJETIVO)


def test_detecta_nulos(df_valido):
    df_valido.loc[0, "LIMIT_BAL"] = None
    problemas = validar_calidad(df_valido, OBJETIVO)
    assert any("LIMIT_BAL" in p and "nulos" in p for p in problemas)


@pytest.mark.parametrize(
    "columna, valor",
    [
        ("LIMIT_BAL", 0),  # el límite de crédito debe ser positivo
        ("AGE", 5),  # menor de edad
        ("AGE", 150),  # fuera de dominio
        ("PAY_0", 15),  # más de 9 meses de atraso no existe en el dataset
        ("PAY_AMT1", -100),  # un pago no puede ser negativo
    ],
)
def test_detecta_valores_fuera_de_rango(df_valido, columna, valor):
    df_valido.loc[0, columna] = valor
    problemas = validar_calidad(df_valido, OBJETIVO)
    assert any(columna in p for p in problemas)


@pytest.mark.parametrize("columna, valor", [("SEX", 3), ("EDUCATION", 9), ("MARRIAGE", 7)])
def test_detecta_categorias_no_permitidas(df_valido, columna, valor):
    df_valido.loc[0, columna] = valor
    problemas = validar_calidad(df_valido, OBJETIVO)
    assert any(columna in p and "no permitidos" in p for p in problemas)


def test_detecta_objetivo_no_binario(df_valido):
    df_valido.loc[0, OBJETIVO] = 2
    problemas = validar_calidad(df_valido, OBJETIVO)
    assert any(OBJETIVO in p for p in problemas)


def test_cargar_datos_rechaza_dataset_invalido(tmp_path):
    ruta = tmp_path / "corrupto.csv"
    df = pd.read_csv(FIXTURE)
    df.loc[0, "AGE"] = -3
    df.to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="calidad"):
        cargar_datos(ruta, OBJETIVO)
