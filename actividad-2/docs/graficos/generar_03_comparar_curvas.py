"""Compara la curva ROC de 2 modelos reales distintos -- para mostrar que el umbral NO cambia
la forma, pero el modelo SI.

Ejecucion (desde la raiz de actividad-2):
    poetry run python docs/graficos/generar_03_comparar_curvas.py
"""
import sys
from pathlib import Path

RAIZ_ACTIVIDAD_2 = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ_ACTIVIDAD_2))

import matplotlib.pyplot as plt
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

AZUL = "#2a78d6"
NARANJA = "#eb6834"
GRIS_BASELINE = "#c3c2b7"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
SUPERFICIE = "#fcfcfb"

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor(SUPERFICIE)
ax.set_facecolor(SUPERFICIE)
ax.plot([0, 1], [0, 1], color=GRIS_BASELINE, linewidth=1.5, linestyle="--", zorder=2)

modelos = [
    ("logistic-regression", AZUL, "Regresión logística\n(sin balancear)"),
    ("random-forest-balanced", NARANJA, "Random forest\n(balanceado, max_depth=8)"),
]
posiciones_label = {
    "logistic-regression": (0.55, 0.48),
    "random-forest-balanced": (0.35, 0.90),
}
for nombre_run, color, etiqueta in modelos:
    candidato = next(c for c in cfg["candidatos"] if c["nombre_run"] == nombre_run)
    modelo = construir_modelo(candidato["tipo"], candidato["parametros"], numericas, categoricas)
    modelo.fit(X_train, y_train)
    proba = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, zorder=3)
    idx = len(fpr) // 2
    xt, yt = posiciones_label[nombre_run]
    ax.annotate(
        f"{etiqueta}\nAUC = {auc:.2f}",
        xy=(fpr[idx], tpr[idx]),
        xytext=(xt, yt),
        color=color,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color=color, linewidth=1),
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Tasa de falsos positivos", fontsize=10.5, color=TINTA_SECUNDARIA)
ax.set_ylabel("Tasa de verdaderos positivos = RECALL", fontsize=10.5, color=TINTA_SECUNDARIA)
ax.set_title("Dos modelos distintos = dos curvas distintas\n(el umbral no tiene nada que ver con esto)", fontsize=13, color=TINTA_PRIMARIA, fontweight="bold", pad=14)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRIS_BASELINE)
ax.tick_params(colors=TINTA_SECUNDARIA)
ax.grid(color="#e1e0d9", linewidth=0.6, zorder=0)

plt.tight_layout()
salida = Path(__file__).resolve().parent / "03-comparar-curvas-modelos.png"
plt.savefig(salida, facecolor=SUPERFICIE)
print(f"Guardado en: {salida}")
