# Actividad 1: Fundamentos, versionamiento y tracking

**Caso de uso:** predicción de incumplimiento de pago (default) de clientes de tarjeta de crédito, a partir de su historial de pagos, límite de crédito y estados de cuenta — relevante para riesgo de crédito y cuentas por cobrar.

**Dataset:** [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI, Taiwan). 30,000 clientes, 23 variables (límite de crédito, historial de pagos `PAY_0`-`PAY_6`, montos de factura `BILL_AMT1`-`6`, pagos `PAY_AMT1`-`6`, demográficos) + columna objetivo `default.payment.next.month`. Descargar el CSV y colocarlo en `data/credit_card_default.csv` (requiere cuenta gratuita de Kaggle; no se versiona en git). Detalle completo de cada columna: [`docs/diccionario-datos.md`](./docs/diccionario-datos.md).

## Estructura

```
actividad-1/
├── configs/config.yaml   # datos, modelo y MLflow/Prefect (sin valores en el código)
├── docs/diccionario-datos.md   # qué es cada columna del dataset
├── flows/                # flow de Prefect: cargar datos → entrenar → resumen
├── src/
│   ├── config.py         # carga de configs/config.yaml
│   ├── datos.py          # carga + validación de columnas + agrupa categorías sin documentar
│   ├── modelo.py          # fábrica de modelos (logistic_regression, random_forest)
│   └── modelado.py        # partición estratificada y métricas binarias
├── notebooks/            # exploración
├── tests/                # pruebas unitarias (usan tests/fixtures/credit_sample.csv, no el dataset real)
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

- [x] Definir caso de uso propio (default de tarjeta de crédito).
- [x] Carga y validación del dataset real en `src/` (con pruebas unitarias).
- [ ] Flow de Prefect (`flows/experimento_flow.py`).
- [ ] Tracking con MLflow.
- [ ] Memoria 1 (plantilla del curso).

## Nota sobre el baseline

Un `LogisticRegression` sin escalar features da `ConvergenceWarning` (las columnas `BILL_AMT*`/`PAY_AMT*` están en una escala de miles/millones frente a `SEX`/`EDUCATION`). Queda pendiente decidir si se agrega un `StandardScaler` o se usa un modelo que no lo necesite (ej. `random_forest`).
