# Actividad 2: Pipeline y empaquetado

**Estado:** solo estructura y configuración, la lógica de los pipelines está pendiente (ver [`PLAN.md`](./PLAN.md)).

**Caso de uso:** el mismo de actividad 1 — predicción de incumplimiento de pago (default) de clientes de tarjeta de crédito. Reutiliza el dataset en `../actividad-1/data/credit_card_default.csv`.

## Qué agrega esta actividad sobre la 1

- Pipeline de **entrenamiento** (ingesta → entrenar → evaluar → registrar) con promoción a `champion` en el Model Registry de MLflow.
- Pipeline de **inferencia** (carga el `champion`, puntúa un lote, guarda predicciones), con despliegue diario en Prefect.
- **API con FastAPI** que sirve el `champion`.
- Selección de variables configurable (`configs/features.yaml`: `ALL` / `NUMERIC` / `CUSTOM`).

## Estructura

```
actividad-2/
├── configs/
│   ├── config.yaml        # datos, modelo, MLflow, umbral de champion, inferencia
│   └── features.yaml      # selección de variables
├── domain/                # lógica de negocio (pipeline + features), sin Prefect/MLflow
├── infraestructure/
│   ├── flows/              # entrenamiento_flow.py, inferencia_flow.py, desplegar.py
│   └── api/                 # main.py (FastAPI)
├── src/                    # utilidades compartidas (config, carga de datos)
├── data/                   # salida de predicciones de inferencia (gitignored)
├── tests/
├── poetry.lock
└── pyproject.toml
```

## Entorno con Poetry

```powershell
poetry install
```

## Pendiente

Ver [`PLAN.md`](./PLAN.md) para las decisiones de diseño abiertas antes de escribir los flows, el dominio y la API.
