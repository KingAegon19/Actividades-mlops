"""Barrido de niveles de drift: ¿cómo responden las tres señales de monitoreo a medida que el drift
emulado crece de 0.0 a 1.0? Para cada nivel arma el lote (domain.monitoreo.construir_lote_con_drift)
y mide, contra el champion real:

- Drift de features: fracción de variables con drift según Evidently y según las pruebas
  estadísticas propias (KS / chi-cuadrado + PSI, domain/pruebas_drift.py).
- Drift de evento: tasa de default del lote y p-valor del chi-cuadrado contra el entrenamiento.
- Rendimiento: AUC del champion sobre el lote, contra su AUC de holdout.

Guarda la tabla (docs/graficos/barrido_drift.csv) y el gráfico (docs/graficos/05-barrido-drift.png)
y muestra el primer nivel en que se dispara cada alerta -- es decir, cuánto drift hace falta para
que el monitoreo lo note. No escribe nada en MLflow ni en Prefect: solo lee el champion.

Requisitos: MLflow levantado con un champion (MLFLOW_TRACKING_URI, o el de configs/config.yaml).
Ejecución (desde la raíz de actividad-3):
    poetry run python scripts/barrido_drift.py
"""
import os
import sys
from pathlib import Path

RAIZ_ACTIVIDAD_3 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_ACTIVIDAD_3))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from mlflow.tracking import MlflowClient

from domain.ingenieria import agregar_variables_derivadas
from domain.modelo import OBJETIVO, VARIABLES, puntuar_lote
from domain.monitoreo import (
    construir_lote_con_drift,
    dividir,
    drift_de_target,
    evaluar_rendimiento,
    fraccion_desde_evidently,
)
from domain.pruebas_drift import drift_por_variable, fraccion_con_drift
from src.config import cargar_config
from src.datos import cargar_datos

NIVELES = [round(n, 1) for n in np.arange(0.0, 1.01, 0.1)]
CARPETA_SALIDA = RAIZ_ACTIVIDAD_3 / "docs" / "graficos"

# Paleta (dataviz, references/palette.md)
AZUL, NARANJA = "#2a78d6", "#eb6834"
GRIS_BASELINE, GRIS_MUTED = "#c3c2b7", "#898781"
TINTA_PRIMARIA, TINTA_SECUNDARIA, SUPERFICIE = "#0b0b0b", "#52514e", "#fcfcfb"


def medir_niveles(cfg):
    df = cargar_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"])
    entrenamiento, _ = dividir(df, cfg["division"])
    referencia = agregar_variables_derivadas(entrenamiento)
    mon = cfg["monitoreo"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    nombre = cfg["registro"]["nombre_modelo"]
    champion = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    auc_referencia = float(MlflowClient().get_run(champion.run_id).data.metrics["auc_holdout"])
    print(f"Champion v{champion.version}, AUC de holdout de referencia = {auc_referencia:.4f}")

    filas = []
    for nivel in NIVELES:
        lote = construir_lote_con_drift(df, cfg["division"], cfg["inferencia"]["tamano_lote"], nivel)

        evidently = Report([DataDriftPreset()]).run(reference_data=referencia[VARIABLES], current_data=lote[VARIABLES])
        tabla = drift_por_variable(referencia, lote, mon["alfa_variable"], mon["umbral_psi"])
        evento = drift_de_target(referencia[OBJETIVO], lote[OBJETIVO], mon["alfa_evento"])
        proba = puntuar_lote(modelo, lote, cfg["metricas"]["umbral"])["probabilidad_default"]
        rendimiento = evaluar_rendimiento(
            lote[OBJETIVO], proba, cfg["metricas"]["umbral"], auc_referencia, mon["tolerancia_auc"]
        )

        filas.append(
            {
                "nivel_drift": nivel,
                "fraccion_evidently": fraccion_desde_evidently(evidently.dict()),
                "fraccion_estadistica": fraccion_con_drift(tabla),
                "psi_maximo": float(tabla["psi"].max()),
                "tasa_default_lote": evento["tasa_lote"],
                "p_valor_evento": evento["p_valor"],
                "alerta_evento": evento["drift"],
                "auc_lote": rendimiento["auc"],
                "recall_lote": rendimiento["recall"],
                "caida_auc": rendimiento["caida_auc"],
                "alerta_rendimiento": rendimiento["degradacion"],
            }
        )
        print(f"nivel {nivel:.1f}: {filas[-1]['fraccion_evidently']:.0%} Evidently | {filas[-1]['fraccion_estadistica']:.0%} pruebas")

    resultados = pd.DataFrame(filas)
    resultados["alerta_features"] = resultados["fraccion_evidently"] > mon["umbral_drift"]
    return resultados, auc_referencia, float(entrenamiento[OBJETIVO].mean())


def primer_nivel(resultados, columna):
    niveles = resultados.loc[resultados[columna], "nivel_drift"]
    return f"{niveles.iloc[0]:.1f}" if len(niveles) else "nunca"


def graficar(resultados, cfg, auc_referencia, tasa_entrenamiento):
    mon = cfg["monitoreo"]
    plt.rcParams["figure.dpi"] = 150
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.4))
    fig.patch.set_facecolor(SUPERFICIE)
    x = resultados["nivel_drift"]

    def estilo(ax, titulo, ylabel):
        ax.set_facecolor(SUPERFICIE)
        ax.set_title(titulo, fontsize=10.5, loc="left", color=TINTA_PRIMARIA, fontweight="bold")
        ax.set_xlabel("nivel de drift emulado", fontsize=9, color=TINTA_SECUNDARIA)
        ax.set_ylabel(ylabel, fontsize=9, color=TINTA_SECUNDARIA)
        for borde in ("top", "right"):
            ax.spines[borde].set_visible(False)
        for borde in ("left", "bottom"):
            ax.spines[borde].set_color(GRIS_BASELINE)
        ax.grid(axis="y", color=GRIS_BASELINE, alpha=0.4, linewidth=0.6)
        ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=8.5)
        ax.set_xticks([0.0, 0.3, 0.6, 1.0])

    # 1) Drift de features: dos métodos independientes
    ax = ejes[0]
    estilo(ax, "Features: sube la fracción con drift", "% de variables con drift")
    ax.plot(x, resultados["fraccion_evidently"] * 100, color=AZUL, marker="o", markersize=5, linewidth=2, label="Evidently")
    ax.plot(x, resultados["fraccion_estadistica"] * 100, color=NARANJA, marker="s", markersize=5, linewidth=2, label="KS / chi² + PSI")
    ax.axhline(mon["umbral_drift"] * 100, color=GRIS_MUTED, linestyle="--", linewidth=1)
    ax.text(1.0, mon["umbral_drift"] * 100 - 5, f"umbral de alerta ({mon['umbral_drift']:.0%})", fontsize=8, color=TINTA_SECUNDARIA, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    # 2) Rendimiento
    ax = ejes[1]
    estilo(ax, "Rendimiento: cae el AUC", "AUC del champion en el lote")
    ax.plot(x, resultados["auc_lote"], color=AZUL, marker="o", markersize=5, linewidth=2)
    ax.axhline(auc_referencia, color=GRIS_MUTED, linestyle=":", linewidth=1)
    ax.text(1.0, auc_referencia + 0.004, f"AUC de holdout del champion ({auc_referencia:.3f})", fontsize=8, color=TINTA_SECUNDARIA, ha="right")
    limite = auc_referencia - mon["tolerancia_auc"]
    ax.axhline(limite, color=GRIS_MUTED, linestyle="--", linewidth=1)
    ax.text(1.0, limite + 0.004, f"umbral de alerta (-{mon['tolerancia_auc']})", fontsize=8, color=TINTA_SECUNDARIA, ha="right")

    # 3) Evento
    ax = ejes[2]
    estilo(ax, "Evento: sube la tasa de default", "tasa de default del lote (%)")
    ax.plot(x, resultados["tasa_default_lote"] * 100, color=AZUL, marker="o", markersize=5, linewidth=2)
    ax.axhline(tasa_entrenamiento * 100, color=GRIS_MUTED, linestyle=":", linewidth=1)
    ax.text(1.0, tasa_entrenamiento * 100 + 0.6, f"tasa en entrenamiento ({tasa_entrenamiento:.1%})", fontsize=8, color=TINTA_SECUNDARIA, ha="right")

    fig.tight_layout()
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_SALIDA / "05-barrido-drift.png"
    fig.savefig(ruta, facecolor=SUPERFICIE)
    print(f"Gráfico guardado en {ruta}")


def main():
    cfg = cargar_config()
    resultados, auc_referencia, tasa_entrenamiento = medir_niveles(cfg)

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta_csv = CARPETA_SALIDA / "barrido_drift.csv"
    resultados.to_csv(ruta_csv, index=False)
    print(f"Tabla guardada en {ruta_csv}")

    print("\nPrimer nivel de drift que dispara cada alerta:")
    print(f"  features   : {primer_nivel(resultados, 'alerta_features')}")
    print(f"  evento     : {primer_nivel(resultados, 'alerta_evento')}")
    print(f"  rendimiento: {primer_nivel(resultados, 'alerta_rendimiento')}")

    graficar(resultados, cfg, auc_referencia, tasa_entrenamiento)


if __name__ == "__main__":
    main()
