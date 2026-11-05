"""Actividad 2: pipeline de inferencia.

Pasos: cargar el modelo champion -> tomar un lote de clientes -> predecir -> guardar CSV ->
registrar el run en MLflow.

El modelo se carga por el alias `champion` (no por número de versión), así que cuando
`entrenamiento_flow.py` mueve el alias a un modelo nuevo, esta inferencia lo usa
automáticamente sin tocar código. El run de inferencia guarda en sus tags la versión
usada y el run de entrenamiento del que viene, para poder trazar una predicción hasta
el modelo exacto que la generó.

Requisitos: servicios levantados, un champion ya registrado (correr antes
infraestructure/flows/entrenamiento_flow.py) y las variables de entorno:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000
    PYTHONUTF8=1

Ejecución (desde la raíz de actividad-2):
    poetry run python infraestructure/flows/inferencia_flow.py
"""
import os
from datetime import date
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task

from src.config import cargar_config
from src.datos import cargar_datos


@task
def cargar_champion(cfg):
    """Carga el modelo con el alias champion. Devuelve el modelo, su versión y su run de entrenamiento."""
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Champion cargado: {nombre} versión {version.version} (run de entrenamiento {version.run_id})")
    return modelo, version.version, version.run_id


@task
def ingerir_lote(cfg):
    """Simula clientes nuevos: muestra aleatoria del CSV completo, con el mismo preprocesado que entrenamiento."""
    df = cargar_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"])
    return df.sample(n=cfg["inferencia"]["tamano_lote"])


@task
def predecir(modelo, lote, cfg):
    """Calcula la probabilidad de default y guarda el resultado en un CSV con la fecha de hoy."""
    # Usa exactamente las columnas con las que se entrenó el champion (según el conjunto de
    # features vigente en ese momento), nunca todas las del CSV -- evita errores si cambia features.yaml.
    umbral = cfg["metricas"]["umbral"]
    proba = modelo.predict_proba(lote[list(modelo.feature_names_in_)])[:, 1]
    resultado = lote.copy()
    resultado["probabilidad_default"] = proba
    resultado["prediccion"] = (proba >= umbral).astype(int)

    carpeta = Path(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)
    print(f"{len(proba)} predicciones guardadas en {ruta} (probabilidad media = {proba.mean():.4f})")
    # Solo valores simples (no arreglos/DataFrames) para que Prefect los pase bien entre tareas.
    return str(ruta), len(proba), float(proba.mean())


@task
def registrar_inferencia(version, run_id_entrenamiento, ruta, filas, proba_media):
    """Crea un run de inferencia con tags que lo conectan a la versión del modelo usado."""
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id_entrenamiento", run_id_entrenamiento)
        mlflow.log_metric("filas_predichas", filas)
        mlflow.log_metric("probabilidad_media", proba_media)
        mlflow.log_artifact(ruta)


@flow(name="inferencia-credit-default", log_prints=True)
def inferencia_flow():
    cfg = cargar_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])

    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingerir_lote(cfg)
    ruta, filas, proba_media = predecir(modelo, lote, cfg)
    registrar_inferencia(version, run_id_entrenamiento, ruta, filas, proba_media)


if __name__ == "__main__":
    inferencia_flow()
