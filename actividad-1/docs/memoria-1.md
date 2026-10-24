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
- Logs del run `prueba` en Prefect, con la línea `Suma de 1 a 5 = 15`: [`capturas/prefect-prueba-logs.png`](./capturas/prefect-prueba-logs.png)
- Run `prueba-mlflow` dentro de `prueba-validacion`, con el parámetro `n=5` y la métrica `suma=15`: [`capturas/mlflow-prueba-mlflow-run.png`](./capturas/mlflow-prueba-mlflow-run.png)

**Evidencia adicional (opcional) de la pipeline del caso de uso**, flow `actividad-1-credit-default` en Prefect / experimento `actividad-1` en MLflow:

- Grafo del flow en Prefect (3 tareas completadas): [`capturas/prefect-experimento-grafo.png`](./capturas/prefect-experimento-grafo.png)
- Logs del flow, con `Filas cargadas: 30000` y el inicio del resumen: [`capturas/prefect-experimento-logs-1.png`](./capturas/prefect-experimento-logs-1.png)
- Logs del flow, con las 5 métricas del resumen: [`capturas/prefect-experimento-logs-2.png`](./capturas/prefect-experimento-logs-2.png)
- Overview del run `orderly-fox-388` en MLflow (experimento `actividad-1`): [`capturas/mlflow-experimento-overview.png`](./capturas/mlflow-experimento-overview.png)
- Métricas del run (`accuracy`, `auc`, `precision`, `recall`, `f1`): [`capturas/mlflow-experimento-metricas.png`](./capturas/mlflow-experimento-metricas.png)
- Parámetros del run (`modelo=logistic_regression`, `n_filas=30000`, etc.) y modelo registrado como artifact: [`capturas/mlflow-experimento-parametros.png`](./capturas/mlflow-experimento-parametros.png)

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

Aprendimos que Poetry es una forma de instalar y gestionar las dependencias del proyecto, que compite con pip, y se vincula a un entorno virtual (como un venv) donde se guardan las dependencias al montar un proyecto. También entendimos la diferencia entre Prefect y MLflow: Prefect **orquesta** la ejecución de los pasos del proceso (carga de datos, entrenamiento, resumen), permitiendo ver en qué paso ocurrió un error y con qué logs; MLflow, en cambio, **registra** los resultados de esas ejecuciones (parámetros, métricas, el modelo entrenado), permitiendo comparar experimentos. Son complementarios: uno ejecuta y vigila el proceso, el otro guarda la memoria de los resultados.

Más que todo, aprendimos el potencial de implementar MLOps: así como existe DevOps (desarrollo enfocado a operaciones), MLOps aplica esa misma idea a los modelos de machine learning — no solo se ejecutan, también se controlan y se monitorean, lo que da la tranquilidad de que el sistema funciona correctamente y no "a la deriva".

Como limitación, lo construido es básico, propio de un proyecto estructurado por estudiantes que aún no tienen experticia, pero que recreamos el proceso que pide el profesor a detalle, buscando el aprendizaje y la estructuración correcta.

Con más tiempo, habríamos implementado una mayor variedad de modelos y comparativas entre ellos, usando MLflow para monitorear varios modelos al mismo tiempo sobre la misma data, de modo que se pudiera identificar mejor cuál es el más adecuado. Es uno de los puntos más interesantes de MLflow: permite comparar experimentos de forma ordenada en vez de hacerlo manualmente.

## 8. Uso de herramientas de IA

Usamos Claude Code (Anthropic) como herramienta de IA a lo largo de la actividad. Le delegamos principalmente la arquitectura técnica, la ejecución de comandos de Poetry, y la configuración/conexión de Docker — sobre todo para evitar repetitividad en los comandos, dado que no teníamos experticia previa en estas herramientas y buscamos apoyo de la IA para usarlas correctamente.

Las decisiones estructurales y de entendimiento del proyecto fueron nuestras; a la IA le pedimos principalmente la creación de carpetas, la ejecución de comandos, y la configuración de Poetry y Docker para el despliegue.

Corregimos principalmente la estructura y el despliegue correcto de Prefect y MLflow, para que funcionaran bien y evitar fallas. Un ejemplo concreto: en un momento avanzamos directo a implementar la pipeline completa del caso de uso (el paso 4 de la guía, que es opcional), sin antes escribir y validar `flows/prueba_flow.py` (el paso 3, que sí es obligatorio). Al revisar la guía del profesor con más detalle nos dimos cuenta de ese salto y lo corregimos antes de dar la actividad por cerrada.

Podemos explicar sin la IA la arquitectura de un proyecto de MLOps: que un modelo necesita una parte de monitoreo (logs, que permiten ver su ejecución y tener control), y con MLflow la conexión directa a los resultados del modelo, con entendimiento en tiempo real.
