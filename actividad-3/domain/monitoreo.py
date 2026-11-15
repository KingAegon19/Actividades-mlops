"""Actividad 3: lógica de drift (emulación y detección). No usa Prefect ni MLflow, para poder
probarse sin servicios levantados -- igual que domain/modelo.py.

El drift se emula sobre las variables numéricas crudas (domain.modelo.NUMERICAS_CRUDAS), antes
de la ingeniería de variables. Así, las 8 variables derivadas quedan afectadas de forma
consistente (se recalculan a partir de las crudas ya con drift), en vez de desplazarlas por
separado como si fueran independientes.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from domain.modelo import NUMERICAS_CRUDAS, OBJETIVO

# Subida máxima de la tasa de default que puede introducir el drift (nivel=1.0 → +15 puntos).
# El dataset tiene ~22% de default; +15 puntos es un cambio fuerte pero no absurdo para una
# emulación (no lleva la tasa a un extremo poco realista como pasaría con +30-40 puntos).
SUBIDA_TASA_MAXIMA = 0.15


def dividir(df, division):
    """Parte los datos igual que el entrenamiento (mismo test_size/random_state de config.yaml).
    Devuelve (entrenamiento, prueba) -- la referencia de drift es siempre el entrenamiento real."""
    return train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )


def aplicar_drift(df, nivel, semilla=42):
    """Emula drift en un lote crudo (antes de la ingeniería de variables): desplaza la
    distribución de una fracción de las variables numéricas crudas y sube la tasa de default.

    `nivel` va de 0.0 (sin drift) a 1.0 (máximo). Con nivel=0.0 el lote queda idéntico.
    """
    if not 0 <= nivel <= 1:
        raise ValueError(f"el nivel de drift debe estar entre 0 y 1, no {nivel}")

    out = df.copy()
    rng = np.random.default_rng(semilla)

    # 1. Drift de features: desplaza la media (2 desviaciones típicas) de una fracción de las
    # variables numéricas crudas, elegidas al azar según el nivel.
    n = round(nivel * len(NUMERICAS_CRUDAS))
    if n > 0:
        columnas_afectadas = rng.choice(NUMERICAS_CRUDAS, size=n, replace=False)
        for columna in columnas_afectadas:
            desviacion = out[columna].std()
            if pd.notna(desviacion) and desviacion > 0:
                out[columna] = out[columna] + 2 * desviacion

    # 2. Drift de evento (target): sube la tasa de default en proporción al nivel.
    out[OBJETIVO] = _subir_tasa(out[OBJETIVO], nivel * SUBIDA_TASA_MAXIMA, rng)
    return out


def _subir_tasa(objetivo, subida, rng):
    """Convierte algunos 0 en 1 (al azar) hasta que la tasa de default sube `subida` puntos."""
    objetivo_int = objetivo.astype(int)
    tasa_actual = objetivo_int.mean()
    nueva_tasa = min(tasa_actual + subida, 1.0)

    positivos_objetivo = round(nueva_tasa * len(objetivo_int))
    faltan = positivos_objetivo - int(objetivo_int.sum())
    if faltan <= 0:
        return objetivo

    indices_ceros = objetivo_int.index[objetivo_int == 0]
    faltan = min(faltan, len(indices_ceros))
    if faltan == 0:
        return objetivo

    subidos = objetivo.copy()
    a_cambiar = rng.choice(indices_ceros, size=faltan, replace=False)
    subidos.loc[a_cambiar] = 1
    return subidos


def fraccion_desde_evidently(resultado):
    """Saca la fracción de variables con drift del resultado de Evidently (`reporte.dict()`)."""
    for metrica in resultado["metrics"]:
        if metrica["metric_name"].startswith("DriftedColumnsCount"):
            return float(metrica["value"]["share"])
    raise ValueError("el resultado de Evidently no incluye DriftedColumnsCount")


def validar_lote(lote):
    """Falla con un mensaje claro si el lote no trae la variable objetivo (sin ella no se
    puede calcular el drift de evento -- en producción, las etiquetas reales llegan con
    retraso, así que un lote puede no tenerlas todavía)."""
    if OBJETIVO not in lote.columns:
        raise ValueError(f"el lote no tiene la columna '{OBJETIVO}': no se puede calcular el drift de evento")
    if lote[OBJETIVO].isna().any():
        raise ValueError(f"el lote tiene '{OBJETIVO}' sin valor: no se puede calcular el drift de evento")


def drift_de_target(referencia, lote, alfa):
    """Compara la tasa de default entre referencia (entrenamiento) y lote con un test
    chi-cuadrado de independencia sobre la tabla de contingencia (default sí/no x conjunto)."""
    ref_int = referencia.astype(int)
    lote_int = lote.astype(int)

    positivos_ref = int(ref_int.sum())
    negativos_ref = len(ref_int) - positivos_ref
    positivos_lote = int(lote_int.sum())
    negativos_lote = len(lote_int) - positivos_lote

    _, p_valor, _, _ = chi2_contingency([[positivos_ref, negativos_ref], [positivos_lote, negativos_lote]])
    return {
        "tasa_referencia": positivos_ref / len(ref_int),
        "tasa_lote": positivos_lote / len(lote_int),
        "p_valor": float(p_valor),
        "drift": bool(p_valor < alfa),
    }
