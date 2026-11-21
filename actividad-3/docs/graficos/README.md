# Gráficos de actividad 3

## `05-barrido-drift.png` y `barrido_drift.csv`

Barrido de niveles de drift emulado (0.0 a 1.0, de 0.1 en 0.1) contra el champion real, generado por
[`../../scripts/barrido_drift.py`](../../scripts/barrido_drift.py). Muestra cómo responden las tres
señales de monitoreo:

1. **Features:** % de variables con drift según Evidently y según las pruebas propias (KS / chi-cuadrado
   + PSI), contra el umbral de alerta (30%). Ambos métodos coinciden casi punto a punto.
2. **Rendimiento:** AUC del champion sobre el lote, contra su AUC de holdout y el umbral de alerta
   (-0.05). Cae hasta ~0.52 (el modelo ya no distingue nada) con drift máximo.
3. **Evento:** tasa de default del lote, contra la del entrenamiento (22.1%).

El CSV trae todas las columnas de la corrida (p-valores, recall, PSI máximo, banderas de alerta).
Los números e interpretación están en [`../../PLAN.md`](../../PLAN.md), sección "Barrido de niveles".

Para regenerarlo (necesita MLflow levantado con un champion):

```powershell
poetry run python scripts/barrido_drift.py
```
