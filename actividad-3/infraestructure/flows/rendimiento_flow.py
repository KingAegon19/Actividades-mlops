"""Actividad 3: seguimiento del rendimiento del champion en producción.

Los otros dos pipelines de monitoreo miran los datos (drift de features) y el resultado (drift del
target). Este mira lo que de verdad importa: ¿el modelo sigue funcionando? Puntúa el lote que dejó
la inferencia (`monitoreo.lote_actual`) y, como el lote trae las etiquetas reales, calcula el AUC y
el recall de ese lote. Lo compara contra el AUC de holdout con el que el champion fue promovido
(guardado en MLflow, en el run de entrenamiento del que viene) y alerta si cae más de
`monitoreo.tolerancia_auc`.

En producción las etiquetas reales llegan con retraso (se sabe si el cliente incumplió meses
después), así que este pipeline corre sobre lotes ya "cerrados", no sobre la predicción del día.

Registra un run en MLflow con tag fase=monitoreo, para ver la evolución del rendimiento en el tiempo.

Requisitos: servicios levantados, un champion registrado, y el lote que deja inferencia_flow.py.
Ejecución (desde la raíz de actividad-3):
    poetry run python infraestructure/flows/rendimiento_flow.py
"""
import os
from datetime import date

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from prefect import flow, task

from domain.modelo import OBJETIVO, puntuar_lote
from domain.monitoreo import evaluar_rendimiento, validar_lote
from infraestructure.flows.inferencia_flow import cargar_champion
from src.config import cargar_config


@task
def auc_de_referencia(run_id_entrenamiento):
    """AUC de holdout del champion: lo que el modelo rendía cuando se promovió."""
    return float(MlflowClient().get_run(run_id_entrenamiento).data.metrics["auc_holdout"])


@task
def medir_rendimiento(modelo, ruta_lote, auc_referencia, cfg):
    """Puntúa el lote y compara su AUC con el de referencia. Falla con un mensaje claro si el
    lote no trae etiquetas (sin ellas no hay forma de medir el rendimiento)."""
    lote = pd.read_csv(ruta_lote)
    validar_lote(lote)
    resultado = puntuar_lote(modelo, lote, cfg["metricas"]["umbral"])
    medicion = evaluar_rendimiento(
        lote[OBJETIVO],
        resultado["probabilidad_default"],
        cfg["metricas"]["umbral"],
        auc_referencia,
        cfg["monitoreo"]["tolerancia_auc"],
    )
    print(
        f"AUC del lote {medicion['auc']:.4f} vs referencia {medicion['auc_referencia']:.4f} "
        f"(caída {medicion['caida_auc']:+.4f}); recall del lote {medicion['recall']:.4f}"
    )
    return medicion


@task
def alerta(medicion, tolerancia):
    """Emite una alerta (log) si el AUC del lote cayó más que la tolerancia."""
    if medicion["degradacion"]:
        print(f"ALERTA: el AUC cayó {medicion['caida_auc']:.4f} (tolerancia {tolerancia})")
    else:
        print(f"Sin alerta: el AUC cayó {medicion['caida_auc']:+.4f} (tolerancia {tolerancia})")
    return medicion["degradacion"]


@task
def registrar_monitoreo(version, nivel_drift, medicion):
    """Guarda la medición en MLflow para poder ver el rendimiento a lo largo del tiempo."""
    with mlflow.start_run(run_name=f"monitoreo-rendimiento-{date.today().isoformat()}"):
        mlflow.set_tag("fase", "monitoreo")
        mlflow.set_tag("tipo_monitoreo", "rendimiento")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("nivel_drift", nivel_drift)
        mlflow.set_tag("alerta_rendimiento", medicion["degradacion"])
        mlflow.log_metric("auc_lote", medicion["auc"])
        mlflow.log_metric("recall_lote", medicion["recall"])
        mlflow.log_metric("auc_referencia", medicion["auc_referencia"])
        mlflow.log_metric("caida_auc", medicion["caida_auc"])


@flow(name="rendimiento-credit-default", log_prints=True)
def rendimiento_flow():
    cfg = cargar_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    auc_referencia = auc_de_referencia(run_id_entrenamiento)
    medicion = medir_rendimiento(modelo, cfg["monitoreo"]["lote_actual"], auc_referencia, cfg)
    hay_alerta = alerta(medicion, cfg["monitoreo"]["tolerancia_auc"])
    registrar_monitoreo(version, cfg["inferencia"]["nivel_drift"], medicion)
    return hay_alerta


if __name__ == "__main__":
    rendimiento_flow()
