# Actividad 3: Monitoreo, drift y CI

**Caso de uso:** el mismo de actividades 1 y 2 — predicción de incumplimiento de pago (default) de clientes de tarjeta de crédito. Copia propia del dataset en `data/credit_card_default.csv` (no versionado en git, igual que en las actividades anteriores).

## Qué agrega esta actividad sobre la 2

Reutiliza el pipeline de entrenamiento/inferencia/API y la ingeniería de variables de `actividad-2/` (copiados y adaptados, no importados — cada actividad es autocontenida) y agrega:

- **Drift emulado en la inferencia** (`domain/monitoreo.py`): el lote que puntúa cada corrida sale de la parte de prueba del dataset (nunca del entrenamiento) y se le aplica un drift controlado por `inferencia.nivel_drift` (0.0 a 1.0) antes de la ingeniería de variables, para que las columnas derivadas reflejen el drift igual que pasaría con datos reales.
- **Dos pipelines de drift en Prefect:**
  - `drift_features_flow.py`: compara la distribución de las variables de entrada del lote contra el entrenamiento con Evidently (`DataDriftPreset`). Guarda un reporte HTML por lote en `reports/` (gitignored, se regenera en cada corrida -- ver `docs/evidencia/` para dos reportes reales conservados como prueba) y alerta si la fracción de variables con drift supera `monitoreo.umbral_drift`.
  - `drift_evento_flow.py`: compara la tasa de default del lote contra el entrenamiento con un test chi-cuadrado. Exige que el lote traiga la columna objetivo (rechaza con un mensaje claro si no la trae) y alerta si el p-valor queda por debajo de `monitoreo.alfa_evento`.
- **Tests de la lógica de drift** (`tests/test_monitoreo.py`), sin MLflow ni Prefect.
- **CI opcional** (`../.github/workflows/ci.yml`): corre `poetry run pytest` en `actividad-3/` en cada PR y en cada push a `master`.

Ver [`PLAN.md`](./PLAN.md) para el detalle de las decisiones y los resultados de las corridas reales (incluye un problema real encontrado y corregido: el tamaño de lote).

## Estructura

```
actividad-3/
├── configs/
│   ├── config.yaml        # datos, candidatos, MLflow, champion, inferencia (+ nivel_drift), monitoreo
│   └── features.yaml      # selección de variables (igual que actividad-2)
├── domain/
│   ├── modelo.py           # pipeline, métricas y regla de champion (igual que actividad-2)
│   ├── ingenieria.py        # 8 variables derivadas (igual que actividad-2)
│   └── monitoreo.py         # NUEVO: emulación de drift + detección (sin Prefect ni MLflow)
├── infraestructure/
│   ├── flows/
│   │   ├── entrenamiento_flow.py   # igual que actividad-2
│   │   ├── inferencia_flow.py      # adaptado: lote de prueba + drift emulado + guarda lote_actual
│   │   ├── drift_features_flow.py  # NUEVO
│   │   ├── drift_evento_flow.py    # NUEVO
│   │   └── desplegar.py            # entrenamiento (manual) + inferencia (diaria)
│   └── api/main.py          # igual que actividad-2
├── scripts/explorar_tamano_lote.py   # exploración que llevó a fijar inferencia.tamano_lote=500
├── src/                    # config.py, datos.py (igual que actividad-2)
├── data/                   # dataset + lote_actual.csv + predicciones (gitignored)
├── reports/                # reportes HTML de Evidently (gitignored, se regenera en cada corrida)
├── docs/evidencia/          # dos reportes reales conservados como prueba (nivel 0.0 y 0.6)
├── tests/
└── pyproject.toml
```

## Entorno

Requiere los servicios de MLflow y Prefect levantados (ver `../services/README.md`), un champion ya registrado por `actividad-2/` o por el entrenamiento de esta misma actividad (comparten el mismo modelo `credit-default-modelo` en MLflow), y en cada terminal:

```powershell
poetry install
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONUTF8 = "1"
```

## Cómo correr el proyecto, en orden

```powershell
# 1. (si no hay champion todavía) entrena y registra
poetry run python infraestructure/flows/entrenamiento_flow.py

# 2. Puntúa un lote de prueba con drift emulado (nivel fijo en configs/config.yaml) y lo guarda
poetry run python infraestructure/flows/inferencia_flow.py

# 3. Compara ese lote contra el entrenamiento y alerta si corresponde
poetry run python infraestructure/flows/drift_features_flow.py
poetry run python infraestructure/flows/drift_evento_flow.py

# 4. Despliegues en Prefect (entrenamiento manual, inferencia diaria)
poetry run python infraestructure/flows/desplegar.py

# 5. API
poetry run uvicorn infraestructure.api.main:app --reload
```

Para comparar niveles de drift (0.0, 0.3, 0.6 -- ver criterio de éxito), cambia `inferencia.nivel_drift` en `configs/config.yaml` entre corridas de los pasos 2 y 3.

## Criterio de éxito: verificado en vivo

| `nivel_drift` | % variables con drift | Alerta de features | Tasa de default del lote | p-valor evento | Alerta de evento |
|---|---|---|---|---|---|
| 0.0 | 0% | No | 21.0% (entrenamiento: 22.1%) | 0.5869 | No |
| 0.6 | 61% | **Sí** | 30.0% (entrenamiento: 22.1%) | 0.0000 | **Sí** |

Detalle completo (incluyendo el problema de tamaño de lote y cómo se corrigió) en [`PLAN.md`](./PLAN.md).

## Pendiente

Memoria 3.
