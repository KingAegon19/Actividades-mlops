from pathlib import Path

import pandas as pd
import pytest

from src.datos import agrupar_categorias_sin_documentar, cargar_datos

FIXTURE = Path(__file__).parent / "fixtures" / "credit_sample.csv"
OBJETIVO = "default.payment.next.month"


def test_cargar_datos_descarta_id():
    df = cargar_datos(FIXTURE, OBJETIVO)
    assert "ID" not in df.columns
    assert len(df) == 10


def test_cargar_datos_falla_si_falta_objetivo():
    with pytest.raises(ValueError):
        cargar_datos(FIXTURE, "columna_que_no_existe")


def test_agrupar_categorias_sin_documentar():
    df = pd.DataFrame({"EDUCATION": [0, 5, 6, 1, 2], "MARRIAGE": [0, 1, 2, 3, 0]})
    df_agrupado = agrupar_categorias_sin_documentar(df)

    assert set(df_agrupado["EDUCATION"]) <= {1, 2, 4}
    assert set(df_agrupado["MARRIAGE"]) <= {1, 2, 3}
