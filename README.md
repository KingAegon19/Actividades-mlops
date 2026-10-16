# Actividades MLOps

Repositorio personal para el desarrollo de las actividades y el proyecto de la materia de MLOps.

El contenido compartido por el profesor (solución de referencia, documentación de cada actividad, plantilla de memoria) está clonado por separado en `../actividades` (repo: [santoto-mlops-content](https://github.com/henrycala/santoto-mlops-content)) y **no se copia aquí**: cada actividad en este repo es un caso de uso propio.

## Estructura

```
Actividades-mlops/
├── services/           # MLflow + Prefect (Docker Compose), igual que el curso
├── actividad-1/        # Actividad 1: fundamentos, versionamiento y tracking
└── ...                 # actividad-2, actividad-3, etc.
```

## Servicios (MLflow + Prefect)

Desde `services/`, con Docker Desktop abierto:

```powershell
cd services
docker compose up -d     # MLflow en http://localhost:5000, Prefect en http://localhost:4200
docker compose down      # los detiene (los datos se conservan)
```

Alternativa sin Docker en `services/README.md`.

## Cada actividad

Cada carpeta `actividad-N/` tiene su propio entorno con Poetry:

```powershell
cd actividad-1
poetry install
poetry run pytest -q
```
