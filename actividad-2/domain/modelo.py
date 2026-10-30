"""Lógica de negocio del modelo de default de tarjeta de crédito: variables, pipeline y métricas.

No usa Prefect ni MLflow: se puede probar sin servicios levantados.
Ver ../../actividad-1/docs/diccionario-datos.md para el detalle de cada columna.
"""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
]
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
    else:
        raise ValueError(f"modelo '{tipo}' no soportado")

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricas))
    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])


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
