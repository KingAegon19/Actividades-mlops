"""Diagrama explicativo de cómo se organiza el split train/test y el k-fold dentro del
pipeline de entrenamiento (domain/modelo.py::evaluar_con_kfold, llamado desde
infraestructure/flows/entrenamiento_flow.py::entrenar_y_registrar). Solo para explicar
la arquitectura -- no es parte del pipeline, usa los tamaños reales del dataset.

Ejecucion (desde la raiz de actividad-2):
    poetry run python docs/graficos/generar_04_split_y_kfold.py
"""
import sys
from pathlib import Path

RAIZ_ACTIVIDAD_2 = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ_ACTIVIDAD_2))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from src.config import cargar_config
from src.datos import cargar_datos

plt.rcParams["figure.dpi"] = 150

cfg = cargar_config()
df = cargar_datos(cfg["datos"]["ruta"], cfg["datos"]["objetivo"])
n_total = len(df)
test_size = cfg["division"]["test_size"]
n_test = round(n_total * test_size)
n_train = n_total - n_test
n_splits = cfg["validacion_cruzada"]["n_splits"]
n_val_fold = round(n_train / n_splits)
n_train_fold = n_train - n_val_fold

# -- Paleta (references/palette.md) --
AZUL = "#2a78d6"
AZUL_CLARO = "#9ec5f4"
NARANJA = "#eb6834"
NARANJA_CLARO = "#f5b393"
GRIS_BASELINE = "#c3c2b7"
GRIS_MUTED = "#898781"
VERDE_GOOD = "#0ca30c"
TINTA_PRIMARIA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
SUPERFICIE = "#fcfcfb"

fig, ax = plt.subplots(figsize=(10, 6.5))
fig.patch.set_facecolor(SUPERFICIE)
ax.set_facecolor(SUPERFICIE)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.5)
ax.axis("off")

# Fila 0: dataset completo
ax.add_patch(mpatches.FancyBboxPatch((0.3, 5.3), 9.4, 0.9, boxstyle="round,pad=0.05", facecolor=GRIS_BASELINE, edgecolor="none"))
ax.text(5, 5.75, f"Dataset completo  ({n_total:,} filas)", ha="center", va="center", fontsize=12, fontweight="bold", color=TINTA_PRIMARIA)

# Flecha hacia el split
ax.annotate("", xy=(5, 4.85), xytext=(5, 5.3), arrowprops=dict(arrowstyle="-|>", color=GRIS_MUTED, linewidth=1.5))
ax.text(5.4, 5.08, "train_test_split (una sola vez, random_state fijo,\nestratificado -- igual para los 6 candidatos)",
        ha="left", va="center", fontsize=8.7, color=TINTA_SECUNDARIA, style="italic")

# Fila 1: train 80% / test 20%
ax.add_patch(mpatches.FancyBboxPatch((0.3, 3.75), 6.5, 0.9, boxstyle="round,pad=0.05", facecolor=AZUL_CLARO, edgecolor=AZUL, linewidth=1.5))
ax.text(3.55, 4.2, f"train_df  ({n_train:,} filas, {100*(1-test_size):.0f}%)", ha="center", va="center", fontsize=11, fontweight="bold", color=TINTA_PRIMARIA)

ax.add_patch(mpatches.FancyBboxPatch((7.1, 3.75), 2.6, 0.9, boxstyle="round,pad=0.05", facecolor=NARANJA_CLARO, edgecolor=NARANJA, linewidth=1.5))
ax.text(8.4, 4.2, f"test_df  ({n_test:,}, {100*test_size:.0f}%)", ha="center", va="center", fontsize=10, fontweight="bold", color=TINTA_PRIMARIA)

# Flecha de train hacia k-fold
ax.annotate("", xy=(3.55, 2.95), xytext=(3.55, 3.75), arrowprops=dict(arrowstyle="-|>", color=AZUL, linewidth=1.5))
# Flecha de test hacia holdout (abajo, lado derecho)
ax.annotate("", xy=(8.4, 1.0), xytext=(8.4, 3.75), arrowprops=dict(arrowstyle="-|>", color=NARANJA, linewidth=1.5))

# k-fold: 5 casillas dentro de train
ancho_fold = 6.5 / n_splits
y_fold = 2.05
for i in range(n_splits):
    x0 = 0.3 + i * ancho_fold
    es_validacion = i == 2  # ilustra un fold cualquiera como "validacion"
    color_fill = NARANJA_CLARO if es_validacion else AZUL_CLARO
    color_edge = NARANJA if es_validacion else AZUL
    ax.add_patch(mpatches.FancyBboxPatch((x0 + 0.05, y_fold), ancho_fold - 0.1, 0.8, boxstyle="round,pad=0.03",
                                          facecolor=color_fill, edgecolor=color_edge, linewidth=1.2))
    etiqueta = "validación" if es_validacion else "entrena"
    ax.text(x0 + ancho_fold / 2, y_fold + 0.4, f"fold {i+1}\n{etiqueta}", ha="center", va="center", fontsize=8, color=TINTA_PRIMARIA)

ax.text(3.55, y_fold + 1.0, f"evaluar_con_kfold: StratifiedKFold(n_splits={n_splits}) -- se repite {n_splits} veces,\ncada vez con un fold distinto de validación",
        ha="center", va="bottom", fontsize=9, color=AZUL, fontweight="bold")

ax.annotate("", xy=(3.55, 1.1), xytext=(3.55, 2.05), arrowprops=dict(arrowstyle="-|>", color=AZUL, linewidth=1.5))
ax.add_patch(mpatches.FancyBboxPatch((0.9, 0.3), 5.3, 0.75, boxstyle="round,pad=0.05", facecolor="white", edgecolor=VERDE_GOOD, linewidth=1.8))
ax.text(3.55, 0.675, "auc_kfold_promedio / recall_kfold_promedio\n→ esto decide si el candidato puede ser champion", ha="center", va="center", fontsize=9, fontweight="bold", color=VERDE_GOOD)

# Holdout
ax.add_patch(mpatches.FancyBboxPatch((7.1, 0.3), 2.6, 0.75, boxstyle="round,pad=0.05", facecolor="white", edgecolor=GRIS_MUTED, linewidth=1.5))
ax.text(8.4, 0.675, "auc_holdout\n(solo chequeo de\noverfitting, no decide)", ha="center", va="center", fontsize=8.5, color=GRIS_MUTED)

ax.set_title("Cómo se organiza el split y el k-fold (por cada uno de los 6 candidatos)",
             fontsize=13, color=TINTA_PRIMARIA, fontweight="bold", pad=10)

plt.tight_layout()
salida = Path(__file__).resolve().parent / "04-split-y-kfold.png"
plt.savefig(salida, facecolor=SUPERFICIE)
print(f"train={n_train} ({n_train_fold} entrena + {n_val_fold} valida por fold) | test={n_test}")
print(f"Guardado en: {salida}")
