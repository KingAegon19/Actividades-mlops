# Actividad 2: Pipeline y empaquetado

**Caso de uso:** el mismo de actividad 1 — predicción de incumplimiento de pago (default) de clientes de tarjeta de crédito. Usa su propia copia del dataset en `data/credit_card_default.csv` (no versionado en git, se descarga con la Kaggle CLI igual que en actividad 1 — ver `../actividad-1/README.md`).

## Qué agrega esta actividad sobre la 1

- Pipeline de **entrenamiento** (ingesta → entrenar cada candidato → evaluar → registrar) con promoción a `champion` en el Model Registry de MLflow, usando AUC **y** recall mínimos (ver `PLAN.md`).
- Pipeline de **inferencia** (carga el `champion` por alias, puntúa un lote simulado, guarda predicciones con fecha), desplegado con horario diario en Prefect.
- **API con FastAPI** que sirve el `champion` para predecir un cliente a la vez.
- Selección de variables configurable (`configs/features.yaml`: `ALL` / `NUMERIC` / `CUSTOM`).

**Extras opcionales** (no los pide el enunciado, agregados como práctica -- ver `PLAN.md` para el detalle):
- Ingeniería de variables (`domain/ingenieria.py`): 8 columnas derivadas, filtradas por AUC univariado.
- Validación con k-fold estratificado (5 folds) en vez de un solo split train/test.
- Un tercer algoritmo candidato (XGBoost, con y sin balancear).

## Estructura

```
actividad-2/
├── configs/
│   ├── config.yaml        # datos, candidatos, MLflow, umbrales de champion, inferencia
│   └── features.yaml      # selección de variables (seleccion: ALL/NUMERIC/CUSTOM)
├── domain/                # lógica de negocio: pipeline sklearn, métricas, regla de champion
├── infraestructure/
│   ├── flows/              # entrenamiento_flow.py, inferencia_flow.py, desplegar.py
│   └── api/                 # main.py (FastAPI)
├── domain/ingenieria.py     # extra: 8 variables derivadas (ver PLAN.md)
├── src/                    # config.py (carga YAML), datos.py (carga/valida el CSV)
├── docs/graficos/           # gráficos explicativos de AUC/recall (material de apoyo)
├── data/                   # dataset + salida de predicciones (gitignored)
├── tests/
├── poetry.lock
└── pyproject.toml
```

## Entorno

Requiere los servicios de MLflow y Prefect levantados (ver `../services/README.md`) y, en cada terminal donde se corra un flow o la API:

```powershell
poetry install
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONUTF8 = "1"   # evita un UnicodeEncodeError al imprimir el resumen de MLflow
```

## Cómo correr el proyecto, en orden

```powershell
# 1. Entrena los candidatos y, si alguno cumple los umbrales, lo registra como champion
poetry run python infraestructure/flows/entrenamiento_flow.py

# 2. Puntúa un lote simulado de clientes con el champion vigente
poetry run python infraestructure/flows/inferencia_flow.py

# 3. Registra los dos flows como despliegues en Prefect (entrenamiento manual, inferencia diaria)
poetry run python infraestructure/flows/desplegar.py

# 4. Sirve el champion por API
poetry run uvicorn infraestructure.api.main:app --reload
```

## API: predicción de ejemplo

Con la API corriendo (`http://localhost:8000`), la documentación interactiva con el ejemplo cargado está en `http://localhost:8000/docs`.

Petición (`POST /predict`) con los datos de un cliente real del dataset:

```json
{
  "LIMIT_BAL": 360000.0, "SEX": 1, "EDUCATION": 1, "MARRIAGE": 2, "AGE": 38,
  "PAY_0": -2, "PAY_2": -2, "PAY_3": -2, "PAY_4": -2, "PAY_5": -2, "PAY_6": -2,
  "BILL_AMT1": 0.0, "BILL_AMT2": 0.0, "BILL_AMT3": 0.0, "BILL_AMT4": 0.0, "BILL_AMT5": 0.0, "BILL_AMT6": 0.0,
  "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 0.0, "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0
}
```

Respuesta real (probada contra el champion versión 2, random-forest-balanced con las 8 variables derivadas y k-fold):

```json
{
  "probabilidad_default": 0.6275129964828424,
  "prediccion": 1,
  "version_champion": "2"
}
```

El mismo valor de `probabilidad_default` que arroja la API para este cliente coincide con el que calculó `inferencia_flow.py` por lote — confirma que ambos caminos usan el mismo modelo y el mismo preprocesamiento.

## Pendiente

Ver [`PLAN.md`](./PLAN.md) para las decisiones de diseño y los resultados de las corridas reales. Falta: Memoria 2.
