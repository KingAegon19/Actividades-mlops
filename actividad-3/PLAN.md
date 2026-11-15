# Plan de actividad 3

Referencia: [`../../actividades/docs/actividad-3.md`](../../actividades/docs/actividad-3.md) (guía del profesor).

## Qué pide la actividad

1. Reutilizar el propio pipeline de entrenamiento/inferencia/API (actividad 2).
2. **Drift emulado en la inferencia:** el lote que puntúa sale de la parte de prueba del dataset, con un nivel de drift (0.0 a 1.0) fijo en config, no elegido por quien ejecuta la pipeline. Guardar una copia del lote para los pipelines de drift.
3. **Dos pipelines de drift:** features (Evidently, reporte por lote) y evento/target (compara la tasa de la variable objetivo, rechaza lotes sin etiquetas).
4. **Alerta** si la fracción de variables con drift supera un umbral configurable.
5. **Tests** de datos, modelo, inferencia, API y lógica de drift, sin necesitar MLflow/Prefect levantados. CI opcional.

## Cómo se traduce a nuestro caso de uso

| Pieza del enunciado | En nuestro proyecto |
|---|---|
| Pipeline propio de entrenamiento/inferencia/API | Copiado y adaptado de `actividad-2/` (mismo caso de uso: default de tarjeta de crédito) |
| Drift de features | `domain/monitoreo.py::aplicar_drift` sobre `domain.modelo.NUMERICAS_CRUDAS` (las 20 columnas originales del CSV, antes de la ingeniería de variables) |
| Drift de evento | La variable objetivo es `default.payment.next.month`; se sube su tasa en el lote en proporción al nivel de drift |
| Lote que puntúa | Muestra de la partición de **prueba** (`domain.monitoreo.dividir`, mismo `test_size`/`random_state` que usa el entrenamiento) -- nunca del entrenamiento, para que la comparación de drift sea justa |

## Decisión: dónde se aplica el drift respecto a la ingeniería de variables

El drift se aplica **antes** de `agregar_variables_derivadas` (domain/ingenieria.py), sobre las columnas crudas. Así, las 8 variables derivadas (que se calculan a partir de las crudas) quedan afectadas como consecuencia natural del drift, en vez de desplazarlas por separado como si fueran independientes -- es lo que pasaría con datos reales, donde nadie "inyecta" drift directamente en una variable derivada.

El lote que se guarda para los pipelines de drift (`monitoreo.lote_actual`) es el lote **ya con drift y con las variables derivadas calculadas** -- el mismo que usa `predecir()` para puntuar. Así `drift_features_flow.py` compara exactamente lo que vio el modelo, no una versión distinta.

## Problema real encontrado: tamaño de lote y ruido de muestreo

Con `inferencia.tamano_lote: 100` (el mismo valor que traía actividad-2 para su lote simulado), el pipeline de drift de features marcaba **45% de las variables con drift incluso con `nivel_drift: 0.0`** (sin drift real) -- habría generado una alerta falsa en cada corrida normal.

Causa: comparar un lote de solo 100 filas contra ~24,000 filas de entrenamiento hace que la distancia entre distribuciones (Wasserstein/Jensen-Shannon, que usa Evidently por columna) salga alta solo por el tamaño de muestra, sobre todo en columnas con cola larga (`BILL_AMT*`, `PAY_AMT*`, que tienen outliers de montos grandes). Se probó con [`scripts/explorar_tamano_lote.py`](./scripts/explorar_tamano_lote.py):

| `tamano_lote` | % variables marcadas con drift (nivel_drift=0.0, debería ser ~0%) |
|---|---|
| 100 | 45.16% |
| 300 | 6.45% |
| 500 | 0.00% |
| 800 | 0.00% |
| 1200 | 0.00% |

Se fijó `inferencia.tamano_lote: 500` en `configs/config.yaml`.

## Resultados: verificación en vivo del criterio de éxito

Corridas reales contra el champion vigente (`credit-default-modelo`, versión 2, el mismo registrado en actividad 2), con `tamano_lote=500`:

| `nivel_drift` | % variables con drift (Evidently) | Alerta de features | Tasa de default del lote | Tasa de entrenamiento | p-valor (chi-cuadrado) | Alerta de evento |
|---|---|---|---|---|---|---|
| 0.0 | 0% | No | 21.0% | 22.1% | 0.5869 | No |
| 0.6 | 61% | **Sí** | 30.0% | 22.1% | 0.0000 | **Sí** |

Los dos reportes HTML son distintos entre sí -- confirma el criterio de éxito ("las ejecuciones con nivel 0.0 y 0.6 producen reportes de drift distintos"). Conservados como evidencia en [`docs/evidencia/`](./docs/evidencia/) (`reports/` se regenera en cada corrida y no se versiona).

No se corrió explícitamente el nivel intermedio (0.3) contra el champion para esta tabla; el diseño es monotónico en `nivel_drift` (ver `tests/test_monitoreo.py::test_drift_mas_alto_cambia_mas_variables`), así que un nivel intermedio queda, por construcción, entre los dos extremos documentados.

## Decisiones técnicas adicionales

| Decisión | Alternativas consideradas | Por qué se eligió |
|---|---|---|
| Reutilizar el mismo `nombre_modelo`/experimento de MLflow que actividad-2 (`credit-default-modelo`) | Un modelo/experimento separado para actividad-3 | Es el mismo sistema en producción, ahora con monitoreo encima -- no un proyecto nuevo. El champion de actividad-2 sigue siendo el que se vigila |
| `SUBIDA_TASA_MAXIMA = 0.15` (nivel=1.0 sube la tasa de default hasta 15 puntos) | Un valor más agresivo (ej. 0.3, como en el ejemplo de referencia del profesor) | La tasa real de default es ~22%; +30 puntos la llevaría a más de 50%, un escenario poco realista para una emulación. +15 puntos en el extremo (nivel=1.0) ya genera un drift detectable y claro (ver tabla de resultados con nivel=0.6: +7.9 puntos reales) |
| Drift sobre columnas crudas, no sobre las 8 derivadas | Aplicar drift también directamente a las derivadas | Las derivadas se calculan a partir de las crudas; desplazarlas por separado sería inconsistente (ej. `utilizacion_promedio` quedaría desincronizada de `BILL_AMT*` y `LIMIT_BAL`) |
| El lote de monitoreo es el mismo que puntúa la inferencia (un solo archivo `lote_actual.csv`) | Que cada pipeline de drift tome su propia muestra | Evita que features/evento comparen lotes distintos entre sí o distintos al que de verdad puntuó la API -- la trazabilidad es más simple con una sola fuente |

## Pendiente

- Memoria 3.
- (Opcional) Ejecutar el CI (`../.github/workflows/ci.yml`) en un PR real y confirmar que corre.
