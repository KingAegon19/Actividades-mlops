"""Genera un grafico explicativo de la curva ROC / AUC usando el champion real del proyecto
(random-forest-balanced, actividad-2). Solo para explicar el concepto -- no es parte del pipeline.

Ejecucion (desde la raiz de actividad-2):
    poetry run python docs/graficos/generar_01_curva_roc_auc.py
"""
import sys
from pathlib import Path

RAIZ_ACTIVIDAD_2 = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ_ACTIVIDAD_2))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from domain.modelo import OBJETIVO, construir_modelo, resolver_features
from src.config import cargar_config, cargar_features
from src.datos import cargar_datos

plt.rcParams["figure.dpi"] = 150

cfg = cargar_config()
features_cfg = cargar_features()
df = cargar_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"])

numericas, categoricas = resolver_features("ALL")
variables = numericas + categoricas
division = cfg["division"]
train_df, test_df = train_test_split(
    df, test_size=division["test_size"], random_state=division["random_state"], stratify=df[OBJETIVO]
)
X_train, y_train = train_df[variables], train_df[OBJETIVO]
X_test, y_test = test_df[variables], test_df[OBJETIVO]

# El champion real: random-forest-balanced (ver configs/config.yaml)
candidato_champion = next(c for c in cfg["candidatos"] if c["nombre_run"] == "random-forest-balanced")
modelo = construir_modelo(candidato_champion["tipo"], candidato_champion["parametros"], numericas, categoricas)
modelo.fit(X_train, y_train)
proba = modelo.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, proba)
auc = roc_auc_score(y_test, proba)

# Punto de operacion real: el umbral que usamos en el pipeline (0.5)
umbral_operativo = cfg["metricas"]["umbral"]
idx_umbral = np.argmin(np.abs(thresholds - umbral_operativo))
fpr_op, tpr_op = fpr[idx_umbral], tpr[idx_umbral]

# -- Paleta (references/palette.md) --
AZUL = "#2a78d6"
AZUL_CLARO = "#9ec5f4"
GRIS_BASELINE = "#c3c2b7"
GRIS_MUTED = "#898781"
VERDE_GOOD = "#0ca30c"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
SUPERFICIE = "#fcfcfb"

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(SUPERFICIE)
ax.set_facecolor(SUPERFICIE)

# Area bajo la curva (el "AUC")
ax.fill_between(fpr, tpr, color=AZUL_CLARO, alpha=0.5, zorder=1)

# Linea base: un modelo que adivina al azar (diagonal)
ax.plot([0, 1], [0, 1], color=GRIS_BASELINE, linewidth=1.5, linestyle="--", zorder=2)

# Curva ROC del champion
ax.plot(fpr, tpr, color=AZUL, linewidth=2.5, zorder=3)

# Punto de operacion (umbral = 0.5, el que usa el pipeline)
ax.scatter([fpr_op], [tpr_op], color=VERDE_GOOD, s=70, zorder=4, edgecolor=SUPERFICIE, linewidth=1.5)

# Etiquetas directas (sin leyenda en caja)
idx_mitad = int(len(fpr) * 0.55)
ax.annotate(
    f"Nuestro modelo\n(random-forest-balanced)\nAUC = {auc:.2f}",
    xy=(fpr[idx_mitad], tpr[idx_mitad]),
    xytext=(0.58, 0.74),
    color=AZUL,
    fontsize=11,
    fontweight="bold",
    ha="left",
    arrowprops=dict(arrowstyle="-", color=AZUL, linewidth=1),
)
ax.annotate(
    "Modelo al azar\n(AUC = 0.50)",
    xy=(0.72, 0.72),
    xytext=(0.74, 0.58),
    color=GRIS_MUTED,
    fontsize=10,
    ha="left",
    arrowprops=dict(arrowstyle="-", color=GRIS_MUTED, linewidth=1),
)
ax.annotate(
    f"Umbral que usamos (0.5):\nrecall = {tpr_op:.2f}\nfalsos positivos = {fpr_op:.2f}",
    xy=(fpr_op, tpr_op),
    xytext=(0.26, 0.32),
    color=VERDE_GOOD,
    fontsize=10,
    fontweight="bold",
    ha="left",
    arrowprops=dict(arrowstyle="-", color=VERDE_GOOD, linewidth=1),
)
ax.text(
    0.03, 0.97,
    "Area sombreada = AUC\n(que tan arriba-izquierda\nse mantiene la curva)",
    transform=ax.transAxes,
    fontsize=9.5,
    color=TINTA_SECUNDARIA,
    va="top",
    ha="left",
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Tasa de falsos positivos  (clientes buenos marcados como riesgo, por error)", fontsize=10.5, color=TINTA_SECUNDARIA)
ax.set_ylabel("Tasa de verdaderos positivos = RECALL  (defaults reales que sí atrapamos)", fontsize=10.5, color=TINTA_SECUNDARIA)
ax.set_title("Curva ROC y AUC: el recall ya está dentro de esta curva", fontsize=13.5, color=TINTA_PRIMARIA, fontweight="bold", pad=14)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRIS_BASELINE)
ax.tick_params(colors=TINTA_SECUNDARIA)
ax.grid(color="#e1e0d9", linewidth=0.6, zorder=0)

plt.tight_layout()
salida = Path(__file__).resolve().parent / "01-curva-roc-auc.png"
plt.savefig(salida, facecolor=SUPERFICIE)
print(f"AUC real = {auc:.4f}")
print(f"Guardado en: {salida}")
