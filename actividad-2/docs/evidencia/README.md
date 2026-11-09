# Evidencia del proceso: entrenamiento, registro, despliegue y API

Capturas reales tomadas en vivo (no simuladas) de los servicios corriendo en local, para
documentar el criterio de éxito del enunciado: *"una nueva ejecución del entrenamiento
genera una versión nueva del modelo y el alias champion pasa a apuntar a ella"*. Ver
`../../PLAN.md` para el detalle de las decisiones y resultados que están detrás de cada una.

## Imágenes

1. **`01-mlflow-runs-experimento.png`** — lista de runs del experimento `credit-default-training`
   en MLflow: los 6 candidatos (`logistic-regression`, `random-forest`, `xgboost`, y sus
   variantes balanceadas) con tag `fase=entrenamiento`.
2. **`02-model-registry-v3-temporal.png`** — Model Registry mostrando `@ champion : Version 3`.
   Esta fue una versión de demostración (hiperparámetro `max_depth` subido a 10 a propósito,
   para forzar un reemplazo de champion y capturar el antes/después) que luego se corrigió
   por tener más overfitting que la v2 -- ver punto 3 y `PLAN.md`, sección "chequeo de
   overfitting en la regla de promoción".
3. **`03-model-registry-v2-final.png`** — Model Registry mostrando `@ champion : Version 2`,
   el estado final y correcto: se devolvió el alias a mano después de agregar el chequeo de
   overfitting a `debe_promover_a_champion` (la v3 nunca habría calificado bajo esa regla).
4. **`04-prefect-deployments.png`** — los dos despliegues en Prefect, ambos en estado `Ready`:
   `entrenamiento-credit-default` (sin horario, manual) e `inferencia-diaria` (cron diario,
   "At 06:00 AM every day").
5. **`05-api-docs-predict-request.png`** y **`06-api-docs-predict-response.png`** — la API
   (`/docs`, Swagger) con una predicción real ejecutada en vivo (`POST /predict`) usando el
   cliente de ejemplo documentado. La respuesta (`probabilidad_default: 0.6275129964828424,
   version_champion: "2"`) coincide exactamente con la que calcula `inferencia_flow.py` por
   lote para el mismo cliente -- confirma que ambos caminos usan el mismo modelo y
   preprocesamiento (ver `../../README.md`).
6. **`07-mlflow-inferencia-run.png`** — un run de `inferencia_flow.py` (tag `fase=inferencia`),
   con `modelo_version: 2` y el `run_id` del entrenamiento que generó ese champion -- permite
   rastrear cualquier predicción hasta el modelo exacto que la hizo.
7. **`08-health-api-v2.png`** — `GET /health` respondiendo `version_champion: "2"`, confirmando
   en vivo cuál versión tiene cargada la API en este momento (coincide con el alias `champion`
   de la captura 3).
