"""Actividad 2: pipeline de entrenamiento.

Pasos: ingesta -> entrenar y registrar en MLflow cada candidato de `candidatos` -> evaluar
el mejor contra los umbrales de champion (configs/config.yaml) -> promover si corresponde.

El modelo recibe el alias `champion` solo si su AUC y su recall superan los umbrales
mínimos (`registro.umbral_auc`, `registro.umbral_recall`) y su AUC supera al champion
actual. Si no, el run queda registrado en MLflow pero el alias no cambia.

La lógica de negocio (pipeline sklearn, métricas, regla de promoción) vive en
domain/modelo.py. Este archivo solo orquesta con Prefect y registra en MLflow.

Requisitos: servicios levantados (ver ../../services/README.md) y las variables de entorno:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000
    PYTHONUTF8=1   # evita un UnicodeEncodeError en Git Bash al imprimir el resumen de MLflow

Ejecución (desde la raíz de actividad-2):
    poetry run python infraestructure/flows/entrenamiento_flow.py
"""
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from domain.ingenieria import agregar_variables_derivadas
from domain.modelo import (
    OBJETIVO,
    calcular_metricas,
    construir_modelo,
    elegir_mejor_candidato,
    evaluar_con_kfold,
    resolver_features,
)
from src.config import cargar_config, cargar_features
from src.datos import cargar_datos


def resolver_columnas_features(features_cfg):
    """Traduce `seleccion` (ALL/NUMERIC/CUSTOM) de features.yaml a lo que espera resolver_features."""
    seleccion = features_cfg["seleccion"]
    if seleccion == "ALL":
        return "ALL"
    if seleccion == "NUMERIC":
        return features_cfg["numeric"]
    if seleccion == "CUSTOM":
        return features_cfg["custom"]
    raise ValueError(f"seleccion '{seleccion}' no soportada (usa ALL, NUMERIC o CUSTOM)")


@task
def ingerir_datos(ruta, objetivo):
    """Carga el CSV con src/datos.py (valida columnas, descarta ID, agrupa categorías sin
    documentar) y agrega las variables derivadas de domain/ingenieria.py (extra opcional,
    no lo pide el enunciado)."""
    df = cargar_datos(ruta, objetivo)
    return agregar_variables_derivadas(df)


@task
def entrenar_y_registrar(df, cfg_candidato, cfg):
    """Entrena un candidato, calcula sus métricas y registra el run en MLflow.

    La métrica que decide el champion (metricas["auc"]/["recall"]) es el promedio de
    k-fold (extra opcional, no lo pide el enunciado) sobre el split de train -- una
    estimación más robusta que un solo split. El split de test aparte (holdout) se deja
    como chequeo independiente de overfitting, no decide nada.

    Devuelve (model_uri, metricas) para que el flow decida después si promueve a champion.
    """
    numericas, categoricas = resolver_features(resolver_columnas_features(cfg["features"]))
    variables = numericas + categoricas
    division = cfg["division"]
    cv_cfg = cfg["validacion_cruzada"]
    umbral = cfg["metricas"]["umbral"]
    train_df, test_df = train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        stratify=df[OBJETIVO],
    )
    X_train, y_train = train_df[variables], train_df[OBJETIVO]
    X_test, y_test = test_df[variables], test_df[OBJETIVO]

    pipeline = construir_modelo(cfg_candidato["tipo"], cfg_candidato["parametros"], numericas, categoricas)
    metricas_kfold = evaluar_con_kfold(
        pipeline, X_train, y_train, umbral, n_splits=cv_cfg["n_splits"], random_state=cv_cfg["random_state"]
    )

    # evaluar_con_kfold clona el pipeline en cada fold y nunca toca el original -- sigue
    # sin entrenar aquí, así que este fit es el que produce el modelo que se registra.
    pipeline.fit(X_train, y_train)
    proba_train = pipeline.predict_proba(X_train)[:, 1]
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    metricas_holdout = calcular_metricas(y_test, proba_test, umbral)
    auc_train = float(roc_auc_score(y_train, proba_train))

    metricas = {
        "auc": metricas_kfold["auc_kfold_promedio"],
        "recall": metricas_kfold["recall_kfold_promedio"],
        "auc_kfold_std": metricas_kfold["auc_kfold_std"],
        "recall_kfold_std": metricas_kfold["recall_kfold_std"],
        "auc_holdout": metricas_holdout["auc"],
        "recall_holdout": metricas_holdout["recall"],
        "accuracy_holdout": metricas_holdout["accuracy"],
        "precision_holdout": metricas_holdout["precision"],
        "f1_holdout": metricas_holdout["f1"],
        "auc_train": auc_train,
        # Diferencia grande entre AUC de train y de holdout (ej. > 0.05-0.10) es señal de
        # overfitting: el modelo "memoriza" el conjunto de entrenamiento en vez de aprender
        # un patrón generalizable.
        "auc_train_menos_test": round(auc_train - metricas_holdout["auc"], 4),
    }

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    with mlflow.start_run(run_name=cfg_candidato["nombre_run"]) as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params(
            {
                "modelo": cfg_candidato["tipo"],
                **cfg_candidato["parametros"],
                "semilla": division["random_state"],
                "umbral": umbral,
                "features": cfg["features"]["seleccion"],
                "n_features": len(variables),
                "kfold_n_splits": cv_cfg["n_splits"],
            }
        )
        mlflow.log_metrics(metricas)
        info = mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            signature=infer_signature(X_test, proba_test),
            input_example=X_test.head(3),
            # random_forest guarda árboles de sklearn y xgboost su propio Booster; skops los
            # marca como no confiables por defecto. Son modelos propios entrenados aquí mismo,
            # así que se confía en esos tipos.
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )
        print(
            f"{cfg_candidato['nombre_run']} ({cfg_candidato['tipo']}): "
            f"AUC_kfold={metricas['auc']:.4f}(+-{metricas['auc_kfold_std']:.4f}) "
            f"recall_kfold={metricas['recall']:.4f}(+-{metricas['recall_kfold_std']:.4f}) "
            f"| holdout: AUC={metricas_holdout['auc']:.4f} AUC_train={auc_train:.4f} "
            f"(dif={metricas['auc_train_menos_test']:+.4f}) recall={metricas_holdout['recall']:.4f}"
        )

    return info.model_uri, metricas


@task
def obtener_metricas_champion_actual(cfg):
    """Lee el AUC del champion vigente (si existe) para comparar a cada candidato contra él."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    client = MlflowClient()
    nombre = cfg["registro"]["nombre_modelo"]
    try:
        champion = client.get_model_version_by_alias(nombre, "champion")
    except MlflowException:
        return None  # todavía no existe ningún champion
    return {"auc": client.get_run(champion.run_id).data.metrics["auc"]}


@task
def registrar_champion(model_uri, metricas, cfg):
    """Registra `model_uri` como nueva versión y le mueve el alias champion."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    client = MlflowClient()
    nombre = cfg["registro"]["nombre_modelo"]
    version = mlflow.register_model(model_uri, nombre)
    client.set_registered_model_alias(nombre, "champion", version.version)
    print(f"Versión {version.version} registrada como champion (AUC={metricas['auc']:.4f} recall={metricas['recall']:.4f})")
    return version.version


@flow(name="entrenamiento-credit-default", log_prints=True)
def entrenamiento_flow():
    cfg = cargar_config()
    cfg = {**cfg, "features": cargar_features()}

    df = ingerir_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"])

    resultados = [entrenar_y_registrar(df, candidato, cfg) for candidato in cfg["candidatos"]]

    metricas_champion_actual = obtener_metricas_champion_actual(cfg)
    # Evalúa cada candidato contra las reglas de promoción (no solo el de mejor AUC de la
    # tanda): así un candidato con AUC ligeramente menor pero que sí cumple el recall mínimo
    # no se descarta solo porque otro, que de todos modos no califica, tiene mejor AUC.
    ganador = elegir_mejor_candidato(
        resultados,
        umbral_auc=cfg["registro"]["umbral_auc"],
        umbral_recall=cfg["registro"]["umbral_recall"],
        metricas_champion_actual=metricas_champion_actual,
        umbral_overfitting=cfg["registro"]["umbral_overfitting"],
    )
    if ganador is None:
        print("Ningún candidato de la tanda cumple los umbrales y/o supera al champion actual")
        return

    mejor_uri, mejores_metricas = ganador
    registrar_champion(mejor_uri, mejores_metricas, cfg)


if __name__ == "__main__":
    entrenamiento_flow()
