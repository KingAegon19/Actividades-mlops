"""Actividad 3: pipeline de drift de evento (target).

Compara la tasa de default del lote que puntuó la inferencia (`monitoreo.lote_actual`) contra
la de los datos de entrenamiento, con un test chi-cuadrado. Rechaza lotes sin la columna
objetivo o con valores nulos ahí -- en producción, las etiquetas reales (si el cliente de
verdad incumplió) llegan con retraso, así que un lote puede no tenerlas todavía.

Emite una alerta si el p-valor queda por debajo de `monitoreo.alfa_evento`.

Ejecución (desde la raíz de actividad-3, después de correr inferencia_flow.py):
    poetry run python infraestructure/flows/drift_evento_flow.py
"""
import pandas as pd
from prefect import flow, task

from domain.modelo import OBJETIVO
from domain.monitoreo import dividir, drift_de_target, validar_lote
from src.config import cargar_config
from src.datos import cargar_datos


@task
def referencia(cfg):
    """Tasa de default de los datos de entrenamiento (mismo split que entrenamiento_flow.py)."""
    df = cargar_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"])
    entrenamiento, _ = dividir(df, cfg["division"])
    return entrenamiento[OBJETIVO]


@task
def evaluar_evento(referencia_objetivo, ruta_lote, cfg):
    """Compara la tasa de default del lote con la referencia. Falla con un mensaje claro si
    el lote no trae la columna objetivo con valores."""
    lote = pd.read_csv(ruta_lote)
    validar_lote(lote)
    resultado = drift_de_target(referencia_objetivo, lote[OBJETIVO], cfg["monitoreo"]["alfa_evento"])
    print(
        f"Tasa de default: entrenamiento {resultado['tasa_referencia']:.1%}, lote {resultado['tasa_lote']:.1%} "
        f"(p-valor {resultado['p_valor']:.4f})"
    )
    return resultado


@task
def alerta(resultado, alfa):
    """Emite una alerta (log) si el drift del target supera el nivel de significación."""
    hay_alerta = resultado["drift"]
    if hay_alerta:
        print(f"ALERTA: drift del target (p-valor {resultado['p_valor']:.4f} < {alfa})")
    else:
        print(f"Sin alerta: drift del target (p-valor {resultado['p_valor']:.4f} >= {alfa})")
    return hay_alerta


@flow(name="drift-evento-credit-default", log_prints=True)
def drift_evento_flow():
    cfg = cargar_config()
    resultado = evaluar_evento(referencia(cfg), cfg["monitoreo"]["lote_actual"], cfg)
    return alerta(resultado, cfg["monitoreo"]["alfa_evento"])


if __name__ == "__main__":
    drift_evento_flow()
