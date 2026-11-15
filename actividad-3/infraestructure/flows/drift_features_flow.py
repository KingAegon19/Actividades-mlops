"""Actividad 3: pipeline de drift de features.

Compara la distribución de las variables de entrada del lote que puntuó la inferencia
(`monitoreo.lote_actual`) contra los datos de entrenamiento, con el preset de drift de
Evidently. Guarda un reporte HTML por lote y emite una alerta si la fracción de variables
con drift supera `monitoreo.umbral_drift`.

No necesita MLflow ni etiquetas -- compara solo las variables de entrada (domain.modelo.VARIABLES),
nunca el target (eso lo hace drift_evento_flow.py).

Ejecución (desde la raíz de actividad-3, después de correr inferencia_flow.py):
    poetry run python infraestructure/flows/drift_features_flow.py
"""
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from pathlib import Path
from prefect import flow, task

from domain.ingenieria import agregar_variables_derivadas
from domain.modelo import VARIABLES
from domain.monitoreo import dividir, fraccion_desde_evidently
from src.config import cargar_config
from src.datos import cargar_datos


@task
def referencia(cfg):
    """Variables de entrada de los datos de entrenamiento: la referencia contra la que se
    compara cada lote. Mismo split que usa entrenamiento_flow.py, así que es exactamente lo
    que vio el champion vigente."""
    df = agregar_variables_derivadas(cargar_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"]))
    entrenamiento, _ = dividir(df, cfg["division"])
    return entrenamiento[VARIABLES]


@task
def calcular_drift(datos_referencia, ruta_lote, cfg):
    """Ejecuta el preset de drift de Evidently sobre el lote guardado y guarda el reporte
    HTML. Devuelve la fracción de variables con drift."""
    lote = pd.read_csv(ruta_lote)
    resultado = Report([DataDriftPreset()]).run(reference_data=datos_referencia, current_data=lote[VARIABLES])

    carpeta = Path(cfg["monitoreo"]["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_html = carpeta / f"drift_features_{Path(ruta_lote).stem}.html"
    resultado.save_html(str(ruta_html))
    print(f"Reporte de drift de features guardado en {ruta_html}")

    return fraccion_desde_evidently(resultado.dict())


@task
def alerta(fraccion, umbral):
    """Emite una alerta (log) si la fracción de variables con drift supera el umbral configurado."""
    hay_alerta = fraccion > umbral
    if hay_alerta:
        print(f"ALERTA: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%})")
    else:
        print(f"Sin alerta: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%})")
    return hay_alerta


@flow(name="drift-features-credit-default", log_prints=True)
def drift_features_flow():
    cfg = cargar_config()
    fraccion = calcular_drift(referencia(cfg), cfg["monitoreo"]["lote_actual"], cfg)
    return alerta(fraccion, cfg["monitoreo"]["umbral_drift"])


if __name__ == "__main__":
    drift_features_flow()
