# Actividad 1: Fundamentos, versionamiento y tracking

**Caso de uso:** predicción de incumplimiento de pago (default) de clientes de tarjeta de crédito, a partir de su historial de pagos, límite de crédito y estados de cuenta — relevante para riesgo de crédito y cuentas por cobrar.

**Dataset:** [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI, Taiwan). 30,000 clientes, 23 variables (límite de crédito, historial de pagos `PAY_0`-`PAY_6`, montos de factura `BILL_AMT1`-`6`, pagos `PAY_AMT1`-`6`, demográficos) + columna objetivo `default.payment.next.month`. Descargar el CSV y colocarlo en `data/credit_card_default.csv` (requiere cuenta gratuita de Kaggle; no se versiona en git). Detalle completo de cada columna: [`docs/diccionario-datos.md`](./docs/diccionario-datos.md).

## Estructura

```
actividad-1/
├── configs/config.yaml   # datos, modelo y MLflow/Prefect (sin valores en el código)
├── docs/diccionario-datos.md   # qué es cada columna del dataset
├── flows/
│   ├── prueba_flow.py     # valida que Prefect + MLflow responden (obligatorio, no usa el caso de uso)
│   └── experimento_flow.py   # pipeline del caso de uso: cargar datos → entrenar → resumen
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
2. En PowerShell, define las variables de entorno (una sola vez por terminal):

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

3. **Validación obligatoria de servicios** — corre primero esto para confirmar que Prefect y MLflow responden:

```powershell
poetry run python flows/prueba_flow.py
```

En `localhost:4200/runs` debe aparecer el flow `prueba` en estado `Completed`, con el log `Suma de 1 a 5 = 15`. En `localhost:5000` debe aparecer el experimento `prueba-validacion` con el run `prueba-mlflow` (parámetro `n`, métrica `suma`).

4. **Pipeline del caso de uso** (recomendado, no obligatorio para cerrar la actividad 1):

```powershell
poetry run python flows/experimento_flow.py
```

## Entregables de la actividad 1

Obligatorios:
- [x] Repositorio en GitHub con estructura y README.
- [x] `pyproject.toml` y `poetry.lock` versionados, `poetry install` funcionando en limpio.
- [x] Servicios validados con `flows/prueba_flow.py` (capturas: run `prueba` en Prefect, experimento `prueba-validacion` en MLflow).
- [ ] Memoria 1 (plantilla del curso).
- [ ] Compañeros de equipo agregados como colaboradores (pendiente: aún no se tienen sus usuarios de GitHub).

Recomendados (no obligatorios para cerrar esta actividad):
- [x] Carga de datos real documentada (`src/datos.py`, `docs/diccionario-datos.md`).
- [x] Pipeline de entrenamiento propia (`flows/experimento_flow.py`): cargar → entrenar → resumen, con tracking en MLflow.

## Nota sobre el baseline

Un `LogisticRegression` sin escalar features da `ConvergenceWarning` (las columnas `BILL_AMT*`/`PAY_AMT*` están en una escala de miles/millones frente a `SEX`/`EDUCATION`). Queda pendiente decidir si se agrega un `StandardScaler` o se usa un modelo que no lo necesite (ej. `random_forest`).
