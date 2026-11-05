"""Actividad 2: API de inferencia.

Carga el modelo champion una sola vez, al arrancar la API (por el alias `champion`, no por
número de versión), y expone un único endpoint para predecir con los datos de un cliente.
No usa Prefect ni crea runs de MLflow por cada petición -- eso ya pasó una sola vez, durante
el entrenamiento. Si el champion cambia (una nueva corrida de entrenamiento lo reemplaza), hay
que reiniciar la API para que cargue la versión nueva.

Requisitos: servicios levantados (ver ../../services/README.md), un champion ya registrado
(correr antes infraestructure/flows/entrenamiento_flow.py) y las variables de entorno:
    MLFLOW_TRACKING_URI=http://localhost:5000
    PYTHONUTF8=1

Ejecución (desde la raíz de actividad-2):
    poetry run uvicorn infraestructure.api.main:app --reload

Documentación interactiva, con el ejemplo de predicción: http://localhost:8000/docs
"""
import os
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from src.config import cargar_config

ESTADO = {}


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    cfg = cargar_config()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")

    ESTADO["modelo"] = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    ESTADO["version_champion"] = version.version
    ESTADO["umbral"] = cfg["metricas"]["umbral"]
    print(f"Champion cargado: {nombre} versión {version.version}")
    yield
    ESTADO.clear()


app = FastAPI(
    title="API de predicción de default de tarjeta de crédito",
    description="Sirve el modelo champion registrado en MLflow para predecir, cliente por cliente, "
    "la probabilidad de incumplimiento de pago (default) el mes siguiente.",
    lifespan=ciclo_de_vida,
)


# Las 23 variables con las que se entrena el champion (ver domain/modelo.py -> VARIABLES).
# Ver ../../actividad-1/docs/diccionario-datos.md para el detalle de cada columna.
class ClienteEntrada(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "LIMIT_BAL": 360000.0,
                "SEX": 1,
                "EDUCATION": 1,
                "MARRIAGE": 2,
                "AGE": 38,
                "PAY_0": -2,
                "PAY_2": -2,
                "PAY_3": -2,
                "PAY_4": -2,
                "PAY_5": -2,
                "PAY_6": -2,
                "BILL_AMT1": 0.0,
                "BILL_AMT2": 0.0,
                "BILL_AMT3": 0.0,
                "BILL_AMT4": 0.0,
                "BILL_AMT5": 0.0,
                "BILL_AMT6": 0.0,
                "PAY_AMT1": 0.0,
                "PAY_AMT2": 0.0,
                "PAY_AMT3": 0.0,
                "PAY_AMT4": 0.0,
                "PAY_AMT5": 0.0,
                "PAY_AMT6": 0.0,
            }
        }
    )

    LIMIT_BAL: float = Field(description="Límite de crédito otorgado")
    SEX: int = Field(description="1=hombre, 2=mujer")
    EDUCATION: int = Field(description="1=posgrado, 2=universidad, 3=bachillerato, 4=otros")
    MARRIAGE: int = Field(description="1=casado, 2=soltero, 3=otros")
    AGE: int
    PAY_0: int = Field(description="Estado de pago del mes más reciente (-2..8, ver diccionario de datos)")
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


class PrediccionSalida(BaseModel):
    probabilidad_default: float = Field(description="Probabilidad (0 a 1) de que el cliente incumpla el mes siguiente")
    prediccion: int = Field(description="0 o 1, de aplicar el umbral de clasificación a probabilidad_default")
    version_champion: str = Field(description="Versión del modelo (en el Model Registry de MLflow) que respondió")


@app.post("/predict", response_model=PrediccionSalida)
def predecir(cliente: ClienteEntrada) -> PrediccionSalida:
    modelo = ESTADO["modelo"]
    fila = pd.DataFrame([cliente.model_dump()])[list(modelo.feature_names_in_)]
    proba = float(modelo.predict_proba(fila)[0, 1])
    return PrediccionSalida(
        probabilidad_default=proba,
        prediccion=int(proba >= ESTADO["umbral"]),
        version_champion=str(ESTADO["version_champion"]),
    )


@app.get("/health")
def salud():
    """Confirma que la API está arriba y qué versión del champion tiene cargada."""
    return {"estado": "ok", "version_champion": ESTADO.get("version_champion")}
