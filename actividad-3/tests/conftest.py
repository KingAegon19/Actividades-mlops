"""Fixtures compartidos. Un dataset sintético con la misma estructura que el real (columnas,
tipos y orden de magnitud) para probar la lógica de inferencia y monitoreo sin servicios ni el CSV
completo -- el CSV real no se versiona, así que los tests no pueden depender de él."""
import numpy as np
import pandas as pd
import pytest

from domain.modelo import OBJETIVO


@pytest.fixture
def datos_crudos():
    """4,000 clientes sintéticos con las columnas del CSV (sin ID) y ~22% de default."""
    rng = np.random.default_rng(7)
    n = 4000
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.uniform(10_000, 500_000, n).round(-3),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.integers(1, 5, n),
            "MARRIAGE": rng.integers(1, 4, n),
            "AGE": rng.integers(21, 70, n),
        }
    )
    for columna in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        df[columna] = rng.choice([-2, -1, 0, 1, 2, 3], size=n, p=[0.1, 0.2, 0.5, 0.1, 0.07, 0.03])
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = rng.lognormal(10, 1, n).round(0)
        df[f"PAY_AMT{i}"] = rng.lognormal(7.5, 1.2, n).round(0)
    df[OBJETIVO] = (rng.random(n) < 0.22).astype(int)
    return df
