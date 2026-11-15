"""Actividad 3: pipeline de drift de features.

Compara la distribución de las variables de entrada del lote que puntuó la inferencia
(`monitoreo.lote_actual`) contra los datos de entrenamiento, de dos formas complementarias:

1. Preset de drift de Evidently: reporte HTML por lote. **De aquí sale la alerta** (fracción de
   variables con drift > `monitoreo.umbral_drift`), como pide el enunciado.
2. Pruebas estadísticas propias por variable (domain/pruebas_drift.py): Kolmogorov-Smirnov para
   las continuas, chi-cuadrado para las discretas, y PSI como magnitud. Guarda una tabla CSV
   que explica la alerta ("¿cuáles variables cambiaron y cuánto?") y sirve de contraste
   independiente: si ambos métodos coinciden, la alerta es más confiable.

No necesita MLflow ni etiquetas -- compara solo las variables de entrada (domain.modelo.VARIABLES),
nunca el target (eso lo hace drift_evento_flow.py).

Ejecución (desde la raíz de actividad-3, después de correr inferencia_flow.py):
    poetry run python infraestructure/flows/drift_features_flow.py
"""
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from prefect import flow, task

from domain.ingenieria import agregar_variables_derivadas
from domain.modelo import VARIABLES
from domain.monitoreo import dividir, fraccion_desde_evidently
from domain.pruebas_drift import drift_por_variable, fraccion_con_drift
from src.config import cargar_config
from src.datos import cargar_datos


@task
def referencia(cfg):
    """Variables de entrada de los datos de entrenamiento: la referencia contra la que se
    compara cada lote. Mismo split que usa entrenamiento_flow.py, así que es exactamente lo
    que vio el champion vigente."""
    df = agregar_variables_derivadas(cargar_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"]))
    entrenamiento, _ = dividir(df, cfg["division"])
    return entrenamiento[VARIABLES]


@task
def calcular_drift(datos_referencia, ruta_lote, cfg):
    """Ejecuta el preset de drift de Evidently sobre el lote guardado y guarda el reporte
    HTML. Devuelve la fracción de variables con drift."""
    lote = pd.read_csv(ruta_lote)
    resultado = Report([DataDriftPreset()]).run(reference_data=datos_referencia, current_data=lote[VARIABLES])

    carpeta = Path(cfg["monitoreo"]["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_html = carpeta / f"drift_features_{Path(ruta_lote).stem}.html"
    resultado.save_html(str(ruta_html))
    print(f"Reporte de drift de features guardado en {ruta_html}")

    return fraccion_desde_evidently(resultado.dict())


@task
def pruebas_por_variable(datos_referencia, ruta_lote, cfg):
    """Corre KS / chi-cuadrado / PSI variable por variable, guarda la tabla en CSV y la resume en
    el log. Devuelve la fracción de variables con drift según estas pruebas."""
    monitoreo = cfg["monitoreo"]
    lote = pd.read_csv(ruta_lote)
    tabla = drift_por_variable(datos_referencia, lote, monitoreo["alfa_variable"], monitoreo["umbral_psi"])

    carpeta = Path(monitoreo["carpeta_reportes"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta_csv = carpeta / f"drift_features_{Path(ruta_lote).stem}_variables.csv"
    tabla.to_csv(ruta_csv, index=False)
    print(f"Tabla de pruebas por variable guardada en {ruta_csv}")

    con_drift = tabla[tabla["drift"]]
    if con_drift.empty:
        print("Pruebas estadísticas: ninguna variable con drift")
    else:
        print(f"Pruebas estadísticas: {len(con_drift)} de {len(tabla)} variables con drift. Las que más cambiaron:")
        for _, fila in con_drift.head(5).iterrows():
            print(f"  - {fila['variable']}: PSI={fila['psi']:.2f} ({fila['severidad_psi']}), {fila['prueba']} p={fila['p_valor']:.4f}")
    return fraccion_con_drift(tabla)


@task
def alerta(fraccion, umbral, fraccion_estadistica):
    """Emite una alerta (log) si la fracción de variables con drift (Evidently) supera el umbral.
    Reporta también si las pruebas estadísticas propias llegan a la misma conclusión."""
    hay_alerta = fraccion > umbral
    if hay_alerta:
        print(f"ALERTA: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%})")
    else:
        print(f"Sin alerta: drift de features en {fraccion:.0%} de las variables (umbral {umbral:.0%})")

    coinciden = (fraccion_estadistica > umbral) == hay_alerta
    veredicto = "coinciden con Evidently" if coinciden else "DISCREPAN de Evidently (revisar el reporte)"
    print(f"Pruebas estadísticas: {fraccion_estadistica:.0%} de las variables con drift -> {veredicto}")
    return hay_alerta


@flow(name="drift-features-credit-default", log_prints=True)
def drift_features_flow():
    cfg = cargar_config()
    datos_referencia = referencia(cfg)
    ruta_lote = cfg["monitoreo"]["lote_actual"]
    fraccion = calcular_drift(datos_referencia, ruta_lote, cfg)
    fraccion_estadistica = pruebas_por_variable(datos_referencia, ruta_lote, cfg)
    return alerta(fraccion, cfg["monitoreo"]["umbral_drift"], fraccion_estadistica)


if __name__ == "__main__":
    drift_features_flow()
