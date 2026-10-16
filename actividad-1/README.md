# Actividad 1: Fundamentos, versionamiento y tracking

**Caso de uso:** predicción de incumplimiento de pago (default) de clientes de tarjeta de crédito, a partir de su historial de pagos, límite de crédito y estados de cuenta — relevante para riesgo de crédito y cuentas por cobrar.

**Dataset:** [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI, Taiwan). 30,000 clientes, 23 variables (límite de crédito, historial de pagos `PAY_0`-`PAY_6`, montos de factura `BILL_AMT1`-`6`, pagos `PAY_AMT1`-`6`, demográficos) + columna objetivo `default.payment.next.month`. Descargar el CSV y colocarlo en `data/credit_card_default.csv` (requiere cuenta gratuita de Kaggle; no se versiona en git).

## Estructura

```
actividad-1/
├── configs/config.yaml   # datos, modelo y MLflow/Prefect (sin valores en el código)
├── flows/                # flow de Prefect: cargar datos → entrenar → resumen
├── src/                  # código del caso de uso (carga de datos, modelo, etc.)
├── notebooks/            # exploración
├── tests/                # pruebas de carga de datos
├── poetry.lock
└── pyproject.toml
```

Los servicios (MLflow y Prefect) no están en esta carpeta: se levantan desde [`../services/`](../services/README.md).

## Entorno con Poetry

```powershell
poetry install
poetry run pytest -q
```

## Ejecución

1. Levanta MLflow y Prefect desde `../services/` (`docker compose up -d`).
2. En PowerShell:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
poetry run python flows/experimento_flow.py
```

## Pendiente

- [ ] Definir caso de uso propio (o dataset de Kaggle).
- [ ] `generate_data.py` (si el dataset es simulado) o carga del dataset real en `src/`.
- [ ] Flow de Prefect (`flows/experimento_flow.py`).
- [ ] Tracking con MLflow.
- [ ] Memoria 1 (plantilla del curso).
