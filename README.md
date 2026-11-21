# Actividades MLOps

![Tests actividad 3](https://github.com/KingAegon19/Actividades-mlops/actions/workflows/ci.yml/badge.svg)

Repositorio del equipo para las actividades y el proyecto de la materia de MLOps (Prefect + MLflow).

**Caso de uso:** predicción del incumplimiento de pago (*default*) de clientes de tarjeta de crédito, con el dataset público [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI, 30,000 clientes). Es el mismo en las tres actividades: cada una construye sobre la anterior.

## Integrantes

| Nombre completo | Usuario de GitHub |
|---|---|
| Juan Pablo Velez Hurtado | [KingAegon19](https://github.com/KingAegon19) |
| Jose Fernando Castrillon Alzate | [josefdocas](https://github.com/josefdocas) |
| Emmanuela Jimenez Prado | [Emmaprado](https://github.com/Emmaprado) |

## Actividades

| Actividad | De qué trata | Código | Memoria |
|---|---|---|---|
| **1** | Fundamentos, versionamiento y tracking | [`actividad-1/`](./actividad-1/) | [Memoria 1](./actividad-1/docs/memoria-1.md) |
| **2** | Pipelines de entrenamiento e inferencia, modelo `champion` en MLflow, despliegue en Prefect y API | [`actividad-2/`](./actividad-2/) | [Memoria 2](./actividad-2/docs/memoria-2.md) |
| **3** | Monitoreo: drift de features, de evento y de rendimiento, tests y CI | [`actividad-3/`](./actividad-3/) | [Memoria 3](./actividad-3/docs/memoria-3.md) |

## Cómo se conectan las piezas

```
CSV ──► entrenamiento (Prefect) ──► MLflow: modelo champion ──┬──► inferencia diaria (Prefect)
                                                              └──► API (FastAPI)
                                                                        │
              lote con drift emulado ◄───────────────────────────────────┘ (actividad 3)
                       │
                       ├──► drift de features  (Evidently + KS / chi-cuadrado / PSI)
                       ├──► drift de evento    (chi-cuadrado sobre la tasa de default)
                       └──► rendimiento        (AUC del champion sobre el lote)

Cada cambio de código ──► CI en GitHub Actions (pytest)
```

## Estructura

```
Actividades-mlops/
├── services/           # MLflow + Prefect (Docker Compose), igual que el curso
├── actividad-1/        # fundamentos, versionamiento y tracking
├── actividad-2/        # pipelines, champion, despliegue y API
├── actividad-3/        # monitoreo, drift y CI
└── .github/workflows/  # CI: corre los tests de actividad-3 en cada pull request
```

El contenido compartido por el profesor (solución de referencia, documentación de cada actividad, plantilla de memoria) está clonado por separado en `../actividades` (repo: [santoto-mlops-content](https://github.com/henrycala/santoto-mlops-content)) y **no se copia aquí**: cada actividad en este repo es un caso de uso propio.

## Servicios (MLflow + Prefect)

Desde `services/`, con Docker Desktop abierto:

```powershell
cd services
docker compose up -d     # MLflow en http://localhost:5000, Prefect en http://localhost:4200
docker compose down      # los detiene (los datos se conservan)
```

Alternativa sin Docker en `services/README.md`.

## Cada actividad

Cada carpeta `actividad-N/` es autocontenida y tiene su propio entorno con Poetry (el dataset no se versiona; ver el README de cada una):

```powershell
cd actividad-3
poetry install
poetry run pytest -q
```

## Flujo de trabajo

Los cambios entran por rama propia y *pull request*; `master` no admite push directo. Cada integrante trabajó en su propia rama.
