"""Prueba del endpoint /predict sin depender de MLflow/Prefect: se carga un modelo de
juguete directamente en ESTADO, sin disparar el lifespan (que sí requiere los servicios)."""
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from domain.modelo import VARIABLES
from infraestructure.api.main import ESTADO, app

CLIENTE_EJEMPLO = {
    "LIMIT_BAL": 360000.0, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 38,
    "PAY_0": -2, "PAY_2": -2, "PAY_3": -2, "PAY_4": -2, "PAY_5": -2, "PAY_6": -2,
    "BILL_AMT1": 0.0, "BILL_AMT2": 0.0, "BILL_AMT3": 0.0, "BILL_AMT4": 0.0, "BILL_AMT5": 0.0, "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 0.0, "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0,
}


def _cargar_modelo_de_juguete():
    """Un modelo minimo, entrenado en memoria, solo para probar el endpoint sin servicios."""
    modelo = LogisticRegression()
    X = [[0] * len(VARIABLES), [1] * len(VARIABLES)]
    y = [0, 1]
    modelo.fit(X, y)
    modelo.feature_names_in_ = VARIABLES  # mismo orden que ClienteEntrada
    return modelo


def test_predecir_devuelve_probabilidad_y_prediccion():
    ESTADO["modelo"] = _cargar_modelo_de_juguete()
    ESTADO["version_champion"] = "0-prueba"
    ESTADO["umbral"] = 0.5

    # TestClient sin "with": no dispara el lifespan (que requiere MLflow real).
    client = TestClient(app)
    respuesta = client.post("/predict", json=CLIENTE_EJEMPLO)

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert 0.0 <= cuerpo["probabilidad_default"] <= 1.0
    assert cuerpo["prediccion"] in (0, 1)
    assert cuerpo["version_champion"] == "0-prueba"


def test_predecir_rechaza_cliente_incompleto():
    ESTADO["modelo"] = _cargar_modelo_de_juguete()
    ESTADO["version_champion"] = "0-prueba"
    ESTADO["umbral"] = 0.5

    client = TestClient(app)
    respuesta = client.post("/predict", json={"LIMIT_BAL": 1000.0})

    assert respuesta.status_code == 422  # error de validación de Pydantic, faltan campos
