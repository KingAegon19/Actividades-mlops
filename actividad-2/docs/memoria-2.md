# Memoria 2: Pipelines, entrenamiento automatizado y despliegue

> **Nota para el equipo:** las secciones 1-6 ya están armadas con lo técnico que decidimos y
> verificamos en vivo (ver [`PLAN.md`](../PLAN.md) para el detalle completo). Las secciones
> **7 (Conclusiones)** y **8 (Uso de IA)** quedan marcadas con `[ ]` para que cada quien
> escriba su parte con su propia voz -- son justo las secciones donde se nota el aporte
> individual, que es lo que pide el profesor. Revisen, corrijan lo que no les cuadre, y
> completen esas dos antes de entregar.

## 1. Objetivo

Construir las pipelines de entrenamiento e inferencia del caso de uso (predicción de
incumplimiento de pago de clientes de tarjeta de crédito, ver Memoria 1), conectadas a través
de MLflow mediante el patrón de alias `champion`, y servir el modelo vigente con una API.
Se apoya directamente en la actividad 1 (mismo caso de uso, mismo dataset, mismo código base
de carga de datos) y agrega la capa de orquestación, versionado de modelos y despliegue que
faltaba.

## 2. Descripción del trabajo

1. **Configuración en YAML.** `configs/config.yaml` (datos, candidatos, MLflow, umbrales de
   champion, validación cruzada, inferencia) y `configs/features.yaml` (selección de
   variables: `ALL`/`NUMERIC`/`CUSTOM`) -- nada hardcodeado en el código.

2. **Ingeniería de variables (`domain/ingenieria.py`).** 8 columnas derivadas del historial
   de pago y consumo (razones, interacciones, logaritmos, cuadrados), filtradas por AUC
   univariado real contra el target. Se aplican de forma consistente en entrenamiento,
   inferencia por lote y la API.

3. **Pipeline de entrenamiento (`infraestructure/flows/entrenamiento_flow.py`).** Pasos:
   ingesta → entrenar 6 candidatos (regresión logística, random forest y XGBoost, cada uno
   con y sin balanceo de clases) → validar cada uno con k-fold estratificado (5 folds) →
   elegir el mejor candidato que cumpla los umbrales (AUC, recall, y un chequeo de
   overfitting) → registrar en MLflow y mover el alias `champion` si corresponde.

4. **Pipeline de inferencia (`infraestructure/flows/inferencia_flow.py`).** Carga el modelo
   por el alias `champion` (nunca por número de versión fijo), puntúa un lote simulado de
   clientes nuevos, guarda las predicciones en un CSV con la fecha de ejecución, y registra
   el run en MLflow con tag `fase=inferencia`.

5. **Despliegue (`infraestructure/flows/desplegar.py`).** Registra los dos flows como
   despliegues separados en Prefect: `entrenamiento-credit-default` sin horario (se lanza a
   mano) e `inferencia-diaria` con horario diario (`0 6 * * *`).

6. **API (`infraestructure/api/main.py`).** FastAPI que carga el `champion` una sola vez al
   arrancar (vía `lifespan`) y expone `POST /predict` para un cliente a la vez, con un
   ejemplo documentado en `/docs`.

7. **Pruebas automatizadas.** 39 tests (`tests/`) cubriendo la lógica de negocio
   (`domain/modelo.py`, `domain/ingenieria.py`), la carga de datos y la API -- sin depender
   de que MLflow/Prefect estén levantados.

## 3. Decisiones técnicas

| Decisión | Alternativas consideradas | Por qué se eligió |
|---|---|---|
| AUC **y** recall mínimos para promover a champion (no solo AUC) | Usar solo AUC, como mínimo exige el enunciado | El AUC no depende de un umbral de decisión, pero un modelo con buen AUC puede tener recall pobre en la práctica; en nuestro caso, perder un default real pesa más que una falsa alarma |
| Validar cada candidato con k-fold (5 folds) en vez de un solo split train/test | Un solo split train/test (lo mínimo que exige el enunciado) | Un solo split puede salir "suertudo" o "desafortunado" por azar; el promedio de 5 folds es una estimación más robusta, y es la que decide el champion (el split de test aparte queda solo como chequeo de overfitting) |
| `class_weight`/`scale_pos_weight` balanceado como candidato adicional | Solo mover el umbral de decisión después de entrenar | Balancear during el entrenamiento es una mejora real del modelo (subió el recall de 0.24 a 0.63 en la regresión logística), no solo un ajuste posterior |
| Chequeo de overfitting (`auc_train_menos_test`) como criterio adicional de promoción | Promover solo con AUC y recall, como exige el enunciado | Durante una prueba real, un candidato con más overfitting (`max_depth` más alto) ganó por una diferencia de AUC insignificante (dentro del ruido del k-fold) pero con mucho peor generalización -- se agregó el chequeo para que el sistema mismo lo descarte |
| Evaluar cada candidato contra las reglas de promoción, no solo el de mejor AUC de la tanda | Comparar solo el candidato con mejor AUC de toda la tanda | Con 6 candidatos, el de mejor AUC a veces fallaba el recall mínimo, y la lógica original no evaluaba al segundo mejor -- se corrigió para que cada candidato se evalúe por separado |
| XGBoost como tercer algoritmo candidato | Quedarse solo con regresión logística y random forest | Extra opcional para practicar; en los datos reales no superó a `random-forest-balanced`, pero quedó disponible en el factory de modelos |
| Ingeniería de variables (8 columnas derivadas) | Usar solo las 23 columnas originales (lo que exige el enunciado) | Extra opcional; se filtraron por AUC univariado real (no una prueba estadística, poco útil con ~30k filas) -- 2 de las nuevas variables resultaron más predictivas que cualquier columna original |
| Lote de inferencia simulado con muestra aleatoria del CSV completo | Separar un holdout fijo dedicado a "clientes nuevos" | No hay clientes nuevos reales en un proyecto de curso; una muestra aleatoria en cada corrida simula ese escenario sin inventar un dataset aparte |

## 4. Evidencias

- Repositorio: https://github.com/KingAegon19/Actividades-mlops (rama `feature/actividad-2-scaffold`)
- Decisiones y resultados completos de cada corrida real: [`PLAN.md`](../PLAN.md)
- Capturas del proceso (Model Registry, Prefect, API, MLflow): [`evidencia/`](./evidencia/) -- ver el índice en [`evidencia/README.md`](./evidencia/README.md)
- Gráficos explicativos de AUC/recall/k-fold con el modelo real: [`graficos/`](./graficos/) -- ver [`graficos/README.md`](./graficos/README.md)
- Ejemplo de predicción documentado (API): [`../README.md`](../README.md), sección "API: predicción de ejemplo"

## 5. Resultados

Resultados finales del champion vigente (`random-forest-balanced`, versión 2 en el Model
Registry, 8 variables derivadas + k-fold):

| Métrica | k-fold (decide el champion) | Holdout (solo chequeo) |
|---|---|---|
| AUC | 0.7848 | 0.7749 |
| Recall | 0.6316 | 0.6194 |
| AUC train | -- | 0.8377 |
| Diferencia train-holdout (overfitting) | -- | +0.0628 (sano) |

Comparación de los 6 candidatos entrenados en la misma corrida (AUC k-fold / recall k-fold):

| Candidato | AUC k-fold | Recall k-fold |
|---|---|---|
| logistic-regression | 0.7572 | 0.2978 |
| random-forest | 0.7840 | 0.3696 |
| logistic-regression-balanced | 0.7592 | 0.6097 |
| **random-forest-balanced (champion)** | **0.7848** | **0.6316** |
| xgboost | 0.7823 | 0.3722 |
| xgboost-balanced | 0.7805 | 0.6287 |

`random-forest-balanced` ganó por tener el mejor AUC k-fold entre los candidatos que cumplen
ambos umbrales mínimos (AUC≥0.70, recall≥0.50) y el chequeo de overfitting (≤0.08).

## 6. Problemas encontrados

- **Lógica de selección de champion incompleta.** La primera versión solo evaluaba el
  candidato con mejor AUC de toda la tanda contra las reglas de promoción. Con 6 candidatos,
  esto bloqueó a `random-forest-balanced` (buen recall) porque `random-forest` sin balancear
  (mejor AUC, pero mal recall) "ganaba" la comparación aunque de todos modos no calificaba.
  Se corrigió evaluando cada candidato por separado.
- **Overfitting no penalizado en la regla de promoción.** Al forzar un champion nuevo subiendo
  `max_depth` (de 8 a 10) como prueba, ganó un candidato con mucho más overfitting
  (`auc_train_menos_test`=0.11 vs 0.06) por una diferencia de AUC insignificante. Se agregó un
  umbral de overfitting como criterio adicional de promoción.

  Durante el proceso de entrenamiento y selección de champion, identificamos que el modelo
  que estaba quedando como champion tenía overfitting. Esto era un problema real porque, al
  consumir la API, los usuarios iban a acceder a un modelo que, aunque mostraba buenas
  métricas de entrenamiento, no generalizaba bien y podía devolver resultados poco confiables.
  Para solucionarlo, agregamos una validación adicional al proceso de selección, más estricta:
  descarta un candidato si tiene demasiado overfitting, aunque su AUC sea mejor que el del
  champion actual. Así nos aseguramos de que, cada vez que se reentrene, el modelo que quede
  como champion cumpla con un estándar real de calidad y no gane solo por una métrica aislada.
- **Tipos no confiables al guardar modelos en MLflow.** `mlflow.sklearn.log_model` rechaza por
  defecto los árboles de `random_forest` y el `Booster` de `xgboost` como "tipos no
  confiables" (protección de `skops`). Se resolvió declarándolos explícitamente como
  confiables (`skops_trusted_types`), ya que son modelos propios entrenados en el mismo
  pipeline.
- **`UnicodeEncodeError` en Git Bash.** Mismo problema que en actividad 1 (el emoji que
  imprime MLflow al cerrar un run); se resuelve con `PYTHONUTF8=1` como variable de entorno.
- *(Agreguen aquí los problemas que encuentren al trabajar en las tareas de
  [`PARA_COMPANEROS.md`](../PARA_COMPANEROS.md).)*

## 7. Conclusiones

Lo que más nos costó entender fue cómo se conecta MLflow con la API: al principio no era
claro, hasta que entendimos que todo pasa por HTTP — la API (FastAPI) le pregunta a MLflow,
vía una petición HTTP, cuál es el modelo champion vigente, lo descarga, y lo usa para
predecir. Y lo que más nos sorprendió fue ver cómo encajan Prefect y MLflow en todo el
proceso: Prefect organiza cuándo se ejecuta el entrenamiento y la inferencia diaria, MLflow
guarda los resultados y decide cuál modelo es el champion, y la API lo consulta para que los
usuarios finales puedan usarlo. Entender ese flujo completo nos hizo ver la importancia real
de MLOps: no es solo entrenar un modelo, es tener todo un sistema que permite que proyectos
de machine learning sean confiables, reproducibles y accesibles para que otras personas los
usen con confianza.

- [ ] *(Resto del equipo: qué aprendieron sobre la diferencia entre orquestación (Prefect) y
  tracking (MLflow), y sobre el patrón de alias `champion` para desacoplar entrenamiento de
  consumo.)*
- [ ] *(Qué fue lo más retador de esta actividad para cada quien, y cómo lo resolvieron.)*
- [ ] *(Con más tiempo, qué mejorarían del sistema.)*

## 8. Uso de herramientas de IA

> Esta sección es un requisito del curso y tiene que reflejar la experiencia real de cada
> quien -- no la completa la IA por ustedes.

- [ ] Herramientas utilizadas y para qué tareas.
- [ ] Qué partes del trabajo se hicieron sin IA.
- [ ] Qué salidas de la IA se revisaron, corrigieron o descartaron, y por qué.
- [ ] Contribución individual de cada integrante frente a la asistencia de IA.
