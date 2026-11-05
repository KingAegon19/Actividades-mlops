"""ROC con varios umbrales marcados sobre la misma curva, para visualizar la tabla de ejemplos.

Ejecucion (desde la raiz de actividad-2):
    poetry run python docs/graficos/generar_02_curva_con_umbrales.py
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
from src.config import cargar_config
from src.datos import cargar_datos

plt.rcParams["figure.dpi"] = 150

cfg = cargar_config()
df = cargar_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"])

numericas, categoricas = resolver_features("ALL")
variables = numericas + categoricas
division = cfg["division"]
train_df, test_df = train_test_split(
    df, test_size=division["test_size"], random_state=division["random_state"], stratify=df[OBJETIVO]
)
X_train, y_train = train_df[variables], train_df[OBJETIVO]
X_test, y_test = test_df[variables], test_df[OBJETIVO]

candidato = next(c for c in cfg["candidatos"] if c["nombre_run"] == "random-forest-balanced")
modelo = construir_modelo(candidato["tipo"], candidato["parametros"], numericas, categoricas)
modelo.fit(X_train, y_train)
proba = modelo.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, proba)
auc = roc_auc_score(y_test, proba)
n_defaults = int(y_test.sum())
n_buenos = len(y_test) - n_defaults

AZUL = "#2a78d6"
AZUL_CLARO = "#9ec5f4"
GRIS_BASELINE = "#c3c2b7"
VERDE_GOOD = "#0ca30c"
CRITICO = "#d03b3b"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
SUPERFICIE = "#fcfcfb"

fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor(SUPERFICIE)
ax.set_facecolor(SUPERFICIE)

ax.fill_between(fpr, tpr, color=AZUL_CLARO, alpha=0.45, zorder=1)
ax.plot([0, 1], [0, 1], color=GRIS_BASELINE, linewidth=1.5, linestyle="--", zorder=2)
ax.plot(fpr, tpr, color=AZUL, linewidth=2.5, zorder=3)

puntos = [0.90, 0.70, 0.50, 0.30, 0.15]
offsets = {
    0.90: (0.07, -0.02),
    0.70: (0.08, -0.01),
    0.50: (0.09, 0.03),
    0.30: (0.07, -0.08),
    0.15: (-0.22, -0.04),
}
for u in puntos:
    idx = np.argmin(np.abs(thresholds - u))
    r, f = tpr[idx], fpr[idx]
    es_actual = u == 0.50
    color = VERDE_GOOD if es_actual else TINTA_SECUNDARIA
    tam = 90 if es_actual else 45
    ax.scatter([f], [r], color=color, s=tam, zorder=5, edgecolor=SUPERFICIE, linewidth=1.3)
    dx, dy = offsets[u]
    atrapados = int(round(r * n_defaults))
    falsas = int(round(f * n_buenos))
    texto = f"umbral={u:.2f}{'  (el actual)' if es_actual else ''}\n{atrapados} atrapados, {falsas} falsas alarmas"
    ax.annotate(
        texto,
        xy=(f, r),
        xytext=(f + dx, r + dy),
        color=color,
        fontsize=9,
        fontweight="bold" if es_actual else "normal",
        ha="left" if dx >= 0 else "right",
        arrowprops=dict(arrowstyle="-", color=color, linewidth=1),
    )

ax.annotate(
    f"AUC = {auc:.2f}\n(resume estos 5 puntos\ny todos los demas posibles)",
    xy=(0.5, 0.5),
    xytext=(0.55, 0.14),
    color=AZUL,
    fontsize=10.5,
    fontweight="bold",
    ha="left",
)

ax.set_xlim(-0.02, 1.25)
ax.set_ylim(-0.02, 1.08)
ax.set_xlabel("Tasa de falsos positivos  (clientes buenos marcados como riesgo, por error)", fontsize=10.5, color=TINTA_SECUNDARIA)
ax.set_ylabel("Tasa de verdaderos positivos = RECALL  (defaults reales que sí atrapamos)", fontsize=10.5, color=TINTA_SECUNDARIA)
ax.set_title("Bajar el umbral mueve el punto a lo largo de la MISMA curva", fontsize=13.5, color=TINTA_PRIMARIA, fontweight="bold", pad=14)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRIS_BASELINE)
ax.tick_params(colors=TINTA_SECUNDARIA)
ax.grid(color="#e1e0d9", linewidth=0.6, zorder=0)

plt.tight_layout()
salida = Path(__file__).resolve().parent / "02-curva-con-umbrales.png"
plt.savefig(salida, facecolor=SUPERFICIE)
print(f"Guardado en: {salida}")
