from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODELOS_DISPONIBLES = {
    "logistic_regression": lambda random_state: LogisticRegression(max_iter=1000, random_state=random_state),
    "random_forest": lambda random_state: RandomForestClassifier(random_state=random_state),
}


def obtener_modelo(nombre: str, random_state: int) -> ClassifierMixin:
    if nombre not in MODELOS_DISPONIBLES:
        raise ValueError(f"Modelo '{nombre}' no soportado. Opciones: {list(MODELOS_DISPONIBLES)}")
    return MODELOS_DISPONIBLES[nombre](random_state)
