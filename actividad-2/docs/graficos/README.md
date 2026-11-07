# Gráficos explicativos: AUC y recall

Material de apoyo para entender por qué `entrenamiento_flow.py` usa **AUC + recall** juntos
para decidir el champion (ver `../../PLAN.md`, sección "El mecanismo de promoción a champion").
No es parte del pipeline -- son scripts aparte, generados con los modelos reales del proyecto
(`configs/config.yaml`), para respaldar el proceso de análisis.

## Imágenes

1. **`01-curva-roc-auc.png`** — curva ROC del champion real (`random-forest-balanced`,
   AUC=0.77), con el punto de recall actual (umbral=0.5) marcado sobre la curva. Muestra que
   el recall es un punto *sobre* la misma curva cuya área es el AUC, no una métrica aparte.
2. **`02-curva-con-umbrales.png`** — la misma curva con 5 umbrales distintos marcados
   (0.90, 0.70, 0.50, 0.30, 0.15), con el recall/falsos-positivos/conteos reales en cada uno.
   Muestra que mover el umbral desliza el punto por la curva ya existente -- no la redibuja.
3. **`03-comparar-curvas-modelos.png`** — compara la curva de `logistic-regression` (AUC=0.71)
   contra `random-forest-balanced` (AUC=0.77). Muestra que lo que sí cambia la forma de la
   curva es entrenar un modelo distinto, no el umbral.
4. **`04-split-y-kfold.png`** — cómo se organiza la validación dentro de
   `entrenar_y_registrar` (`entrenamiento_flow.py`): un solo split train/test (80/20,
   24,000/6,000 filas), igual para los 6 candidatos; dentro del 80% de train corre el
   k-fold (5 folds, `domain/modelo.py::evaluar_con_kfold`) y su promedio es lo que decide
   el champion; el 20% de test queda aparte, solo como chequeo de overfitting.

## Cómo regenerarlas

Requieren el dataset en `../../data/credit_card_default.csv` (ver `../../README.md`). Desde
la raíz de `actividad-2`:

```powershell
poetry run python docs/graficos/generar_01_curva_roc_auc.py
poetry run python docs/graficos/generar_02_curva_con_umbrales.py
poetry run python docs/graficos/generar_03_comparar_curvas.py
poetry run python docs/graficos/generar_04_split_y_kfold.py
```
