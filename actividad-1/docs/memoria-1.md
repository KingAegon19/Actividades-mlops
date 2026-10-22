# Memoria 1: Fundamentos, versionamiento y tracking

## 1. Objetivo

Construir la base reproducible del proyecto de la materia: estructura de repositorio, entorno de dependencias fijado, y validación de que los servicios de orquestación (Prefect) y tracking de experimentos (MLflow) funcionan y se conectan entre sí. Es la primera actividad del curso, por lo que no hay actividades previas sobre las que apoyarse; sienta la base para el pipeline y el monitoreo que se construyen en las actividades 2 y 3.

## 2. Descripción del trabajo

1. **Repositorio y entorno.** Se creó el repositorio `Actividades-mlops` en GitHub, con la estructura `configs/`, `data/`, `docs/`, `flows/`, `src/`, `tests/`. El entorno de Python se gestiona con Poetry (`pyproject.toml` + `poetry.lock`), con las dependencias fijadas con versión exacta (`prefect==3.8.8`, `mlflow==3.17.0`, `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.4`, `pyyaml==6.0.3`, y `pytest==9.1.1` como dependencia de desarrollo).

2. **Caso de uso.** Se definió un caso de uso propio: predicción de incumplimiento de pago (default) de clientes de tarjeta de crédito, usando el dataset público [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI, 30,000 filas, 23 variables). Se documentó cada columna en `docs/diccionario-datos.md`.

3. **Código base (`src/`).** `config.py` carga la configuración desde `configs/config.yaml`; `datos.py` carga el CSV, valida sus columnas y agrupa categorías sin documentar (`EDUCATION`, `MARRIAGE`); `modelo.py` es una fábrica de modelos de scikit-learn; `modelado.py` hace la partición estratificada train/test y calcula métricas binarias. Cada módulo tiene pruebas unitarias en `tests/`, usando un fixture pequeño en vez del dataset completo.

4. **Servicios.** MLflow y Prefect se levantan con Docker Compose desde `services/` (`docker compose up -d`), con las imágenes oficiales `ghcr.io/mlflow/mlflow:v3.17.0` y `prefecthq/prefect:3.8.8-python3.12`.

5. **Flows de Prefect.**
   - `flows/prueba_flow.py`: valida que Prefect y MLflow responden. Suma 1 a 5 y registra el resultado como parámetro/métrica en el experimento `prueba-validacion` de MLflow.
   - `flows/experimento_flow.py`: pipeline del caso de uso (cargar datos → entrenar → resumen), con el entrenamiento registrado en MLflow (parámetros, métricas, modelo como artifact).

6. **Flujo de trabajo en Git.** La rama `master` tiene protección (push directo bloqueado); todo cambio entra por rama `feature/...` y Pull Request.

## 3. Decisiones técnicas

| Decisión | Alternativas consideradas | Por qué se eligió |
|---|---|---|
| Dataset real de Kaggle en vez de datos sintéticos (`generate_data.py`) | Generar datos dummy con semilla fija | Un dataset real da resultados y problemas más representativos (ej. desbalance de clases, categorías sin documentar) que un dataset sintético no expondría necesariamente |
| "Default of Credit Card Clients" (30,000 filas) | Dataset de quiebra de empresas ("Company Bankruptcy Prediction", 6,819 filas) | Se descartó el de quiebra por ser demasiado pequeño para que el ejercicio de MLOps se sintiera representativo; ambos son igual de válidos en relevancia temática (riesgo financiero/contable) |
| Poetry para dependencias | pip + `requirements.txt`, Conda | Poetry resuelve y fija versiones exactas en un lockfile, y crea el entorno virtual automáticamente; es la opción que recomienda la guía del curso |
| Agrupar categorías sin documentar (`EDUCATION`/`MARRIAGE` en valores 0, 5, 6) en vez de eliminarlas | Eliminar las filas con esos valores | Son menos del 1.5% de las filas; agruparlas en "otros" conserva el tamaño de muestra sin inventar una categoría |
| Separar `prueba_flow.py` (validación de infraestructura) de `experimento_flow.py` (pipeline del caso de uso) | Un solo flow que hiciera ambas cosas | La guía del curso distingue un paso obligatorio (validar que los servicios funcionan) de uno opcional (la pipeline completa); mantenerlos separados deja claro cuál es cuál |
| No versionar el dataset crudo en git | Versionarlo (la licencia CC0 del dataset lo permitiría) | Se sigue la convención general de no versionar datos descargables, documentando en el README cómo obtenerlos |

## 4. Evidencias

- Repositorio: https://github.com/KingAegon19/Actividades-mlops
- Vista general de runs en Prefect: [`capturas/prefect-runs-overview.png`](./capturas/prefect-runs-overview.png)
- Lista de runs en Prefect, con el run `prueba` (`nostalgic-chachalaca`) en estado `Completed`: [`capturas/prefect-prueba-run-list.png`](./capturas/prefect-prueba-run-list.png)
- Experimentos en MLflow, con `prueba-validacion` creado: [`capturas/mlflow-experimentos.png`](./capturas/mlflow-experimentos.png)
- _Pendiente: captura de los **logs** del run `prueba` en Prefect, donde debe verse la línea `Suma de 1 a 5 = 15`._
- _Pendiente: captura del run `prueba-mlflow` dentro del experimento `prueba-validacion`, con el parámetro `n` y la métrica `suma` visibles._
- _Pendiente (opcional): captura de un run del experimento `actividad-1-credit-default` (pipeline del caso de uso) en MLflow._

## 5. Resultados

Resultados del pipeline del caso de uso (`flows/experimento_flow.py`, `logistic_regression` sobre partición 80/20 estratificada):

| Métrica | Valor |
|---|---|
| Accuracy | 0.8077 |
| AUC | 0.7093 |
| Precision | 0.6821 |
| Recall | 0.2442 |
| F1 | 0.3596 |

El recall es bajo: el modelo detecta pocos de los clientes que realmente incumplen (22% de la muestra). Esto es esperable como primer baseline sin escalado de features ni ajuste de umbral — no es el resultado final del proyecto, solo confirma que el pipeline de principio a fin funciona.

## 6. Problemas encontrados

- **`UnicodeEncodeError` en Git Bash.** Al correr el flow, MLflow imprime un emoji (🏃) al finalizar un run; la consola de Git Bash en Windows usa una codificación (cp1252) que no lo soporta, y el proceso fallaba justo al final (después de haber registrado el run correctamente). Se resolvió corriendo el flow desde PowerShell, que sí maneja UTF-8 sin problema.
- **`ConvergenceWarning` en `LogisticRegression`.** Las columnas de montos (`BILL_AMT*`, `PAY_AMT*`) están en una escala mucho mayor que las demás variables, y el modelo no converge en las 1000 iteraciones por defecto. Queda pendiente decidir si se agrega un `StandardScaler` o se usa un modelo que no lo necesite.
- **Alcance de la actividad.** Inicialmente se implementó directamente la pipeline completa del caso de uso (paso 4 de la guía, opcional) sin antes escribir y validar `flows/prueba_flow.py` (paso 3, obligatorio). Se corrigió al releer la guía de la actividad con detalle.
- **Script de validación no coincidía con el repositorio de referencia.** El `prueba_flow.py` que describe la guía del profesor (con registro en MLflow, experimento `prueba-validacion`) no es igual al que trae el repositorio de referencia del curso (`prueba_prefect_flow.py`, que solo prueba Prefect). Se escribió una versión propia siguiendo la especificación textual de la guía.

## 7. Conclusiones

_Pendiente de completar por el estudiante: qué se aprendió del proceso (ej. diferencia entre orquestación y tracking, importancia de fijar versiones, de separar configuración de código) y qué se mejoraría con más tiempo (ej. escalar features, comparar modelos, agregar al equipo)._

## 8. Uso de herramientas de IA

_Esta sección la debe completar el estudiante con honestidad — es un requisito explícito del curso y no se puede redactar en su nombre. Como guía de lo que se usó en este trabajo:_

- **Herramienta usada:** Claude Code (Anthropic), como asistente conversacional dentro del editor, para investigar, escribir código y ejecutar comandos (git, Docker, Poetry, Kaggle CLI) a lo largo de toda la actividad.
- **Punto a desarrollar:** detallar qué partes revisaste, entendiste y validaste tú mismo (por ejemplo: la elección del caso de uso, la lectura de la guía de la actividad y la corrección del alcance al notar que faltaba `prueba_flow.py`) frente a lo que la IA propuso o escribió directamente.
- **Punto a desarrollar:** qué saliste de la IA tuviste que corregir, cuestionar o rechazar (por ejemplo: en algún momento se avanzó más de lo que pedía la actividad, y fue necesario corregir el alcance).
- **Punto a desarrollar:** qué partes, si las hay, hiciste sin asistencia de IA.
