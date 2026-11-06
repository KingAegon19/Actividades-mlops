"""Ingeniería de variables: columnas derivadas del historial de pago y consumo.

Extra opcional sobre lo que pide el enunciado literal de actividad-2 (solo exige
selección de variables vía configs/features.yaml, no crear variables nuevas). Es una
versión reducida -- proporcional a este dataset de 30k filas y 23 columnas -- de las
familias que se usan en proyectos de detección de fraude a mayor escala: razones,
interacciones, logaritmos, cuadrados y señales de dominio. Cada candidata se filtró
midiendo su AUC univariado real contra el target (en vez de una prueba estadística
tipo Kruskal, que no aporta mucho con tan pocas filas):

| variable                          | AUC discriminativo | ¿se queda? |
|------------------------------------|---------------------|------------|
| interaccion_atraso_x_utilizacion   | 0.710               | sí |
| interaccion_atraso_x_limite        | 0.707               | sí |
| atraso_promedio_cuadrado           | 0.644               | sí |
| log_limit_bal                      | 0.618               | sí |
| razon_pago_reciente                | 0.579               | sí |
| log_bill_promedio                  | 0.519               | no (ruido) |
| bill_en_cero                       | 0.509               | no (ruido) |
| bill_multiplo_1000                 | 0.508               | no (ruido) |

Las señales de dominio "balance en cero / múltiplo de 1000" (típicas para detectar
transacciones de prueba en fraude) no aportan nada aquí -- tiene sentido: son
heurísticas pensadas para otro problema, no para riesgo de default.
"""
import numpy as np

BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_STATUS_COLS = ["PAY_0"] + [f"PAY_{i}" for i in range(2, 7)]

VARIABLES_DERIVADAS = [
    "meses_con_atraso",
    "atraso_promedio",
    "utilizacion_promedio",
    "interaccion_atraso_x_utilizacion",
    "interaccion_atraso_x_limite",
    "atraso_promedio_cuadrado",
    "log_limit_bal",
    "razon_pago_reciente",
]


def agregar_variables_derivadas(df):
    """Agrega las columnas derivadas (ver tabla del módulo) a una copia de `df`.

    - meses_con_atraso: de los 6 meses (PAY_0, PAY_2..PAY_6), cuántos tienen el
      estado de pago en atraso (valor > 0).
    - atraso_promedio: promedio de esos mismos 6 estados de pago (puede ser negativo:
      -2/-1 son "sin consumo"/"pagado completo").
    - utilizacion_promedio: promedio de BILL_AMT1-6 sobre LIMIT_BAL (0 si LIMIT_BAL es 0).
    - interaccion_atraso_x_utilizacion / _x_limite: cuánto pesa el atraso combinado con
      el riesgo de consumo/límite -- son las dos variables con mayor AUC univariado.
    - atraso_promedio_cuadrado: magnitud del atraso sin importar el signo.
    - log_limit_bal: log(1 + LIMIT_BAL) -- suaviza la escala de los límites más altos.
    - razon_pago_reciente: PAY_AMT1 sobre BILL_AMT1 (0 si no hubo facturación ese mes).
    """
    df = df.copy()

    df["meses_con_atraso"] = (df[PAY_STATUS_COLS] > 0).sum(axis=1)
    df["atraso_promedio"] = df[PAY_STATUS_COLS].mean(axis=1)

    limite = df["LIMIT_BAL"].replace(0, np.nan)
    df["utilizacion_promedio"] = (df[BILL_COLS].mean(axis=1) / limite).fillna(0)

    df["interaccion_atraso_x_utilizacion"] = df["meses_con_atraso"] * df["utilizacion_promedio"]
    df["interaccion_atraso_x_limite"] = df["meses_con_atraso"] * df["LIMIT_BAL"]
    df["atraso_promedio_cuadrado"] = df["atraso_promedio"] ** 2
    df["log_limit_bal"] = np.log1p(df["LIMIT_BAL"])

    bill1 = df["BILL_AMT1"].clip(lower=0).replace(0, np.nan)
    df["razon_pago_reciente"] = (df["PAY_AMT1"] / bill1).fillna(0)

    return df
