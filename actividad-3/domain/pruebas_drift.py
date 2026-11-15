"""Pruebas estadísticas de drift por variable. Complementan el preset de Evidently: dan, para cada
variable, un test de hipótesis y una medida de magnitud, de modo que una alerta se pueda explicar
("¿cuáles variables cambiaron y cuánto?") en vez de ser solo un porcentaje.

- Variables continuas: Kolmogorov-Smirnov de dos muestras (compara las distribuciones completas).
- Variables discretas: chi-cuadrado sobre la tabla de contingencia categoría x (referencia, lote).
- PSI (Population Stability Index): magnitud del cambio, la medida estándar en riesgo de crédito.
  Regla práctica: < 0.10 estable, 0.10-0.25 cambio moderado, > 0.25 cambio importante.

No usa Prefect ni MLflow: se puede probar sin servicios levantados.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from domain.ingenieria import PAY_STATUS_COLS
from domain.modelo import CATEGORICAS, VARIABLES

# Variables con pocos valores distintos: un KS no tiene sentido sobre ellas (muchos empates),
# así que se comparan por frecuencia de categorías con chi-cuadrado.
VARIABLES_DISCRETAS = CATEGORICAS + PAY_STATUS_COLS + ["meses_con_atraso"]

PSI_MODERADO = 0.10
PSI_IMPORTANTE = 0.25
# Evita log(0) y división por cero cuando una categoría/intervalo queda vacío en un lado.
EPSILON = 1e-4
# Regla clásica del chi-cuadrado: cada celda debería tener al menos ~5 observaciones esperadas.
MINIMO_ESPERADO = 5


def psi_continua(referencia, lote, n_intervalos=10):
    """PSI de una variable continua: intervalos por cuantiles de la referencia."""
    bordes = np.unique(np.quantile(referencia, np.linspace(0, 1, n_intervalos + 1)))
    if len(bordes) < 3:  # variable casi constante: un solo intervalo, nada que comparar
        return 0.0
    bordes[0], bordes[-1] = -np.inf, np.inf
    frec_ref = np.histogram(referencia, bins=bordes)[0] / len(referencia)
    frec_lote = np.histogram(lote, bins=bordes)[0] / len(lote)
    return _psi(frec_ref, frec_lote)


def psi_discreta(referencia, lote):
    """PSI de una variable discreta: una 'celda' por categoría."""
    categorias = sorted(set(referencia.unique()) | set(lote.unique()))
    frec_ref = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    frec_lote = lote.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    return _psi(frec_ref, frec_lote)


def _psi(frec_ref, frec_lote):
    frec_ref = np.clip(frec_ref, EPSILON, None)
    frec_lote = np.clip(frec_lote, EPSILON, None)
    return float(np.sum((frec_lote - frec_ref) * np.log(frec_lote / frec_ref)))


def severidad_psi(psi):
    if psi >= PSI_IMPORTANTE:
        return "importante"
    if psi >= PSI_MODERADO:
        return "moderado"
    return "estable"


def prueba_ks(referencia, lote):
    """Kolmogorov-Smirnov de dos muestras. Devuelve (estadístico, p-valor)."""
    resultado = ks_2samp(referencia, lote)
    return float(resultado.statistic), float(resultado.pvalue)


def prueba_chi2(referencia, lote):
    """Chi-cuadrado de homogeneidad entre las frecuencias de categorías de referencia y lote.
    Las categorías con menos de MINIMO_ESPERADO observaciones esperadas en el lote se agrupan en
    'otras', para que el test sea válido con categorías raras (ej. PAY_0 = 8 casi no aparece).
    Devuelve (estadístico, p-valor)."""
    categorias = sorted(set(referencia.unique()) | set(lote.unique()))
    c_ref = referencia.value_counts().reindex(categorias, fill_value=0).to_numpy()
    c_lote = lote.value_counts().reindex(categorias, fill_value=0).to_numpy()

    esperado_lote = (c_ref + c_lote) * len(lote) / (len(referencia) + len(lote))
    raras = esperado_lote < MINIMO_ESPERADO
    tabla = np.array([c_ref[~raras], c_lote[~raras]])
    if raras.any():
        tabla = np.column_stack([tabla, [c_ref[raras].sum(), c_lote[raras].sum()]])

    if tabla.shape[1] < 2:  # todo cayó en una sola categoría: no hay nada que comparar
        return 0.0, 1.0
    estadistico, p_valor, _, _ = chi2_contingency(tabla)
    return float(estadistico), float(p_valor)


def evaluar_variable(nombre, referencia, lote, alfa, psi_minimo):
    """Evalúa una variable. Hay drift cuando el test es significativo (p < alfa) **y** el cambio
    es de magnitud relevante (PSI >= psi_minimo): con muestras grandes, un p-valor pequeño solo no
    implica un cambio que importe en la práctica."""
    discreta = nombre in VARIABLES_DISCRETAS
    if discreta:
        prueba, (estadistico, p_valor) = "chi2", prueba_chi2(referencia, lote)
        psi = psi_discreta(referencia, lote)
    else:
        prueba, (estadistico, p_valor) = "ks", prueba_ks(referencia, lote)
        psi = psi_continua(referencia, lote)
    return {
        "variable": nombre,
        "tipo": "discreta" if discreta else "continua",
        "prueba": prueba,
        "estadistico": estadistico,
        "p_valor": p_valor,
        "psi": psi,
        "severidad_psi": severidad_psi(psi),
        "drift": bool(p_valor < alfa and psi >= psi_minimo),
    }


def drift_por_variable(referencia, lote, alfa, psi_minimo, variables=None):
    """Tabla con una fila por variable, ordenada de mayor a menor PSI (la que más cambió, primero)."""
    variables = VARIABLES if variables is None else variables
    filas = [evaluar_variable(v, referencia[v], lote[v], alfa, psi_minimo) for v in variables]
    return pd.DataFrame(filas).sort_values("psi", ascending=False).reset_index(drop=True)


def fraccion_con_drift(tabla):
    """Fracción de variables marcadas con drift en la tabla de `drift_por_variable`."""
    return float(tabla["drift"].mean())
