"""Exploración puntual: ¿qué tamaño de lote hace falta para que Evidently no marque drift por
pura casualidad de muestreo, cuando nivel_drift=0.0 (el lote no debería tener drift real)?

Con lotes chicos, comparar contra ~24,000 filas de referencia hace que la distancia entre
distribuciones (Wasserstein/Jensen-Shannon) salga alta solo por el tamaño de muestra, sobre
todo en columnas con cola larga (BILL_AMT*, PAY_AMT*). Este script prueba varios tamaños y
deja ver en qué punto ese ruido desaparece -- así se fijó `inferencia.tamano_lote` en config.yaml.

Ejecución (desde la raíz de actividad-3):
    poetry run python scripts/explorar_tamano_lote.py
"""
from evidently import Report
from evidently.presets import DataDriftPreset

from domain.ingenieria import agregar_variables_derivadas
from domain.modelo import VARIABLES
from domain.monitoreo import dividir, fraccion_desde_evidently
from src.config import cargar_config
from src.datos import cargar_datos

TAMANOS_A_PROBAR = [100, 300, 500, 800, 1200]


def main():
    cfg = cargar_config()
    df = agregar_variables_derivadas(cargar_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"]))
    entrenamiento, prueba = dividir(df, cfg["division"])
    referencia = entrenamiento[VARIABLES]

    for n in TAMANOS_A_PROBAR:
        lote = prueba.sample(n=n, random_state=cfg["division"]["random_state"])[VARIABLES]
        resultado = Report([DataDriftPreset()]).run(reference_data=referencia, current_data=lote)
        fraccion = fraccion_desde_evidently(resultado.dict())
        print(f"n={n}: fracción con drift (sin drift emulado, debería ser ~0%) = {fraccion:.2%}")


if __name__ == "__main__":
    main()
