import os

import mlflow
from prefect import flow, task

from src.config import cargar_config
from src.datos import agrupar_categorias_sin_documentar, cargar_datos
from src.modelado import calcular_metricas_binarias, partir_datos
from src.modelo import obtener_modelo


@task
def tarea_cargar_datos(config: dict):
    df = cargar_datos(config["datos"]["ruta"], config["datos"]["objetivo"])
    return agrupar_categorias_sin_documentar(df)


@task
def tarea_entrenar(df, config: dict):
    objetivo = config["datos"]["objetivo"]
    X_train, X_test, y_train, y_test = partir_datos(
        df, objetivo, config["modelo"]["test_size"], config["modelo"]["random_state"]
    )

    modelo = obtener_modelo(config["modelo"]["nombre"], config["modelo"]["random_state"])
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    metricas = calcular_metricas_binarias(y_test, y_pred, y_proba)
    return modelo, metricas


@task
def tarea_resumen(metricas: dict):
    print("Resumen del experimento:")
    for nombre, valor in metricas.items():
        print(f"  {nombre}: {valor:.4f}")


@flow(name="actividad-1-credit-default")
def experimento_flow():
    config = cargar_config()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    df = tarea_cargar_datos(config)

    with mlflow.start_run():
        mlflow.log_param("modelo", config["modelo"]["nombre"])
        mlflow.log_param("random_state", config["modelo"]["random_state"])
        mlflow.log_param("test_size", config["modelo"]["test_size"])

        modelo, metricas = tarea_entrenar(df, config)

        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(modelo, name="modelo")

    tarea_resumen(metricas)


if __name__ == "__main__":
    experimento_flow()
