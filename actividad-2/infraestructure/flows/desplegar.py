"""Actividad 2: despliegue de los pipelines en Prefect.

Registra entrenamiento e inferencia como despliegues separados, cada uno con su propio
horario (ver configs/config.yaml -> PLAN.md, sección "Horario de los pipelines"):

- entrenamiento-credit-default: sin horario obligatorio -- se lanza a mano desde la UI
  de Prefect (http://localhost:4200) o con `prefect deployment run`.
- inferencia-diaria: horario diario (cron "0 6 * * *", hora UTC) -- obligatorio según
  el enunciado de la actividad.

Requisitos: servicios levantados (ver ../../services/README.md) y las variables de entorno:
    PREFECT_API_URL=http://localhost:4200/api
    MLFLOW_TRACKING_URI=http://localhost:5000
    PYTHONUTF8=1

Ejecución (desde la raíz de actividad-2). Deja la terminal abierta: mientras esté
corriendo, Prefect puede lanzar los despliegues programados.
    poetry run python infraestructure/flows/desplegar.py
"""
from prefect import serve

from infraestructure.flows.entrenamiento_flow import entrenamiento_flow
from infraestructure.flows.inferencia_flow import inferencia_flow

if __name__ == "__main__":
    despliegue_entrenamiento = entrenamiento_flow.to_deployment(name="entrenamiento-credit-default")
    despliegue_inferencia = inferencia_flow.to_deployment(name="inferencia-diaria", cron="0 6 * * *")
    serve(despliegue_entrenamiento, despliegue_inferencia)
