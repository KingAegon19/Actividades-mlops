"""Lógica de negocio del modelo de default de tarjeta de crédito: variables, pipeline y métricas.

No usa Prefect ni MLflow: se puede probar sin servicios levantados.
Ver ../../actividad-1/docs/diccionario-datos.md para el detalle de cada columna.
"""
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from domain.ingenieria import VARIABLES_DERIVADAS

NUMERICAS = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
] + VARIABLES_DERIVADAS
CATEGORICAS = ["SEX", "EDUCATION", "MARRIAGE"]
VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "default.payment.next.month"


def resolver_features(columnas):
    """Convierte `columnas` de config.yaml ('ALL' o una lista de nombres) en (numericas, categoricas)."""
    if columnas == "ALL":
        elegidas = VARIABLES
    else:
        desconocidas = [v for v in columnas if v not in VARIABLES]
        if desconocidas:
            raise ValueError(f"variables desconocidas en 'columnas': {desconocidas}")
        elegidas = columnas

    if not elegidas:
        raise ValueError("la selección de features no tiene ninguna variable")
    numericas = [v for v in NUMERICAS if v in elegidas]
    categoricas = [v for v in CATEGORICAS if v in elegidas]
    return numericas, categoricas


def construir_modelo(tipo, parametros, numericas, categoricas):
    """Preprocesado (escalar y one-hot) más el estimador elegido en la configuración."""
    if tipo == "logistic_regression":
        estimador = LogisticRegression(**parametros)
    elif tipo == "random_forest":
        estimador = RandomForestClassifier(**parametros)
    elif tipo == "xgboost":
        estimador = XGBClassifier(**parametros)
    else:
        raise ValueError(f"modelo '{tipo}' no soportado")

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricas))
    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


def evaluar_con_kfold(pipeline_sin_entrenar, X, y, umbral, n_splits, random_state):
    """Valida un pipeline (sin entrenar) con k-fold estratificado: en cada fold se clona,
    se entrena con el resto y se evalúa en el fold -- una estimación más robusta que un
    solo split train/test, porque promedia el desempeño sobre varias particiones distintas.

    Devuelve el promedio y la desviación estándar de AUC y recall entre los `n_splits` folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, recalls = [], []
    for indices_train, indices_val in cv.split(X, y):
        modelo = clone(pipeline_sin_entrenar)
        modelo.fit(X.iloc[indices_train], y.iloc[indices_train])
        proba = modelo.predict_proba(X.iloc[indices_val])[:, 1]
        pred = (proba >= umbral).astype(int)
        aucs.append(roc_auc_score(y.iloc[indices_val], proba))
        recalls.append(recall_score(y.iloc[indices_val], pred, zero_division=0))
    return {
        "auc_kfold_promedio": float(np.mean(aucs)),
        "auc_kfold_std": float(np.std(aucs)),
        "recall_kfold_promedio": float(np.mean(recalls)),
        "recall_kfold_std": float(np.std(recalls)),
    }


def elegir_mejor_candidato(resultados, umbral_auc, umbral_recall, metricas_champion_actual):
    """Entre varios candidatos `(model_uri, metricas)` de una misma tanda, devuelve el de
    mejor AUC que además cumpla `debe_promover_a_champion` -- a diferencia de comparar solo
    el candidato con mejor AUC de toda la tanda, así no se descarta uno con buen recall solo
    porque otro (que de todos modos no califica) tiene un AUC un poco más alto.

    Devuelve None si ningún candidato califica.
    """
    validos = [r for r in resultados if debe_promover_a_champion(r[1], umbral_auc, umbral_recall, metricas_champion_actual)]
    if not validos:
        return None
    return max(validos, key=lambda r: r[1]["auc"])


def calcular_metricas(y_true, proba, umbral):
    """Métricas de clasificación binaria a partir de la probabilidad de default."""
    pred = (proba >= umbral).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def debe_promover_a_champion(metricas, umbral_auc, umbral_recall, metricas_champion_actual):
    """Decide si un candidato gana el alias `champion`.

    Exige AUC y recall mínimos a la vez: el AUC filtra que el modelo ordene bien el riesgo
    en general (evita que gane uno que marca a todos como riesgo solo para inflar el recall),
    y el recall mínimo evita que gane un modelo con buen AUC pero que en la práctica casi no
    detecta defaults reales -- en este caso de uso, perder un default pesa más que una falsa alarma.
    """
    if metricas["auc"] < umbral_auc:
        return False
    if metricas["recall"] < umbral_recall:
        return False
    if metricas_champion_actual is not None and metricas["auc"] <= metricas_champion_actual["auc"]:
        return False
    return True
