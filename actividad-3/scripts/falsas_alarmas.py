"""
Experimento T3: evaluación de falsas alarmas y detección de drift.

Se realizan múltiples simulaciones para evaluar el comportamiento del
monitoreo de features bajo dos escenarios:

1. Drift = 0.0: estimación de falsas alarmas.
2. Drift = 0.3: estimación de capacidad de detección.

Los resultados se almacenan para su posterior análisis.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. RUTAS DEL PROYECTO
# ---------------------------------------------------------

RAIZ_ACTIVIDAD_3 = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(RAIZ_ACTIVIDAD_3))

# ---------------------------------------------------------
# 2. IMPORTACIONES DEL PROYECTO
# ---------------------------------------------------------

from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import cargar_config
from src.datos import cargar_datos

from domain.modelo import VARIABLES
from domain.ingenieria import agregar_variables_derivadas
from domain.monitoreo import (
    dividir,
    construir_lote_con_drift,
    fraccion_desde_evidently,
)
# ---------------------------------------------------------
# 3. CONFIGURACIÓN DEL EXPERIMENTO
# ---------------------------------------------------------

NUM_SIMULACIONES = 50

ESCENARIOS_DRIFT = [0.0, 0.3]

SEMILLAS = range(NUM_SIMULACIONES)

# ---------------------------------------------------------
# 4. RUTA DE RESULTADOS
# ---------------------------------------------------------

CARPETA_SALIDA = RAIZ_ACTIVIDAD_3 / "docs" / "graficos"

ARCHIVO_RESULTADOS = (
    CARPETA_SALIDA / "resultados_falsas_alarmas.csv"
)

# ---------------------------------------------------------
# 5. CARGA DE CONFIGURACIÓN Y DATOS
# ---------------------------------------------------------

def preparar_experimento():
    """Carga la configuración y los datos base del experimento."""

    cfg = cargar_config()

    df = cargar_datos(
        cfg["datos"]["ruta"],
        cfg["datos"]["objetivo"]
    )

    return cfg, df


# ---------------------------------------------------------
# 6. PREPARACIÓN DE DATOS POR SIMULACIÓN
# ---------------------------------------------------------

def preparar_datos_simulacion(df, cfg, semilla):
    """Prepara los datos de referencia para una simulación."""

    division = cfg["division"].copy()

    division["random_state"] = semilla

    entrenamiento, prueba = dividir(df, division)

    referencia = agregar_variables_derivadas(entrenamiento)

    return referencia, prueba, division

# ---------------------------------------------------------
# 7. GENERACIÓN DEL LOTE CON DRIFT
# ---------------------------------------------------------

def generar_lote_simulacion(df, division, cfg, nivel_drift):
    """Genera un lote de datos con el nivel de drift especificado."""

    lote = construir_lote_con_drift(
        df,
        division,
        cfg["inferencia"]["tamano_lote"],
        nivel_drift
    )


    return lote

    # ---------------------------------------------------------
# 8. EVALUACIÓN DE DRIFT CON EVIDENTLY
# ---------------------------------------------------------

def evaluar_simulacion(referencia, lote, cfg, nivel_drift, semilla):
    """Evalúa el drift de un lote y determina si genera una alerta."""

    mon = cfg["monitoreo"]

    # Evaluación del drift con Evidently
    evidently = Report([DataDriftPreset()]).run(
        reference_data=referencia[VARIABLES],
        current_data=lote[VARIABLES]
    )

    # Obtener fracción de variables con drift
    fraccion_drift = fraccion_desde_evidently(evidently.dict())

    # Aplicar criterio oficial de alerta
    alerta = fraccion_drift > mon["umbral_drift"]

    # Consolidar resultados
    resultado = {
        "semilla": semilla,
        "nivel_drift": nivel_drift,
        "fraccion_evidently": fraccion_drift,
        "umbral_drift": mon["umbral_drift"],
        "alerta_features": alerta,
    }

    return resultado
# ---------------------------------------------------------
# 9. EXPERIMENTO T3: FALSAS ALARMAS Y DETECCIÓN
# ---------------------------------------------------------

if __name__ == "__main__":

    cfg, df = preparar_experimento()

    resultados = []

    print("\n" + "=" * 60)
    print("EXPERIMENTO T3: FALSAS ALARMAS Y DETECCIÓN")
    print("=" * 60)

    print(f"Simulaciones por escenario: {NUM_SIMULACIONES}")
    print(f"Niveles de drift: {ESCENARIOS_DRIFT}")

    # Recorrer las 50 semillas
    for semilla in SEMILLAS:

        # Preparar referencia para la semilla
        referencia, prueba, division = preparar_datos_simulacion(
            df,
            cfg,
            semilla
        )

        # Evaluar ambos escenarios de drift
        for nivel_drift in ESCENARIOS_DRIFT:

            # Generar lote
            lote = generar_lote_simulacion(
                df,
                division,
                cfg,
                nivel_drift
            )

            # Evaluar simulación
            resultado = evaluar_simulacion(
                referencia,
                lote,
                cfg,
                nivel_drift,
                semilla
            )

            resultados.append(resultado)

        print(f"Simulación {semilla + 1}/{NUM_SIMULACIONES} completada")

    # -----------------------------------------------------
    # 10. CONSOLIDACIÓN DE RESULTADOS
    # -----------------------------------------------------

    resultados_df = pd.DataFrame(resultados)

    # Calcular tasas por escenario
    resumen = (
        resultados_df
        .groupby("nivel_drift")
        .agg(
            simulaciones=("alerta_features", "count"),
            alertas=("alerta_features", "sum"),
            fraccion_media_drift=("fraccion_evidently", "mean"),
            fraccion_mediana_drift=("fraccion_evidently", "median"),
        )
        .reset_index()
    )

    resumen["tasa_alertas"] = (
        resumen["alertas"] / resumen["simulaciones"]
    )

    # -----------------------------------------------------
    # 11. GUARDAR RESULTADOS
    # -----------------------------------------------------

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    resultados_df.to_csv(
        ARCHIVO_RESULTADOS,
        index=False
    )

    archivo_resumen = (
        CARPETA_SALIDA / "resumen_falsas_alarmas.csv"
    )

    resumen.to_csv(
        archivo_resumen,
        index=False
    )

    # -----------------------------------------------------
    # 12. MOSTRAR RESULTADOS
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTADOS DEL EXPERIMENTO T3")
    print("=" * 60)

    print(resumen.to_string(index=False))

    # -----------------------------------------------------
    # 13. INTERPRETACIÓN DE LAS TASAS
    # -----------------------------------------------------

    tasa_falsas_alarmas = resumen.loc[
        resumen["nivel_drift"] == 0.0,
        "tasa_alertas"
    ].iloc[0]

    tasa_deteccion = resumen.loc[
        resumen["nivel_drift"] == 0.3,
        "tasa_alertas"
    ].iloc[0]

    print("\n" + "=" * 60)
    print("INDICADORES PRINCIPALES")
    print("=" * 60)

    print(
        f"Tasa de falsas alarmas: "
        f"{tasa_falsas_alarmas:.2%}"
    )

    print(
        f"Tasa de detección: "
        f"{tasa_deteccion:.2%}"
    )

    print("\nArchivos generados:")
    print(ARCHIVO_RESULTADOS)
    print(archivo_resumen)
