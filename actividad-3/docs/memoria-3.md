# Memoria 3: Monitoreo, drift y CI

> **Nota para el equipo:** las secciones 1-6 ya están armadas con lo técnico que construimos y
> verificamos en vivo (ver [`PLAN.md`](../PLAN.md) para el detalle completo). Las secciones
> **7 (Conclusiones)** y **8 (Uso de IA)** quedan marcadas con `[ ]` y con preguntas guía, para
> que las escriban con su propia voz -- igual que en la Memoria 2. Revisen lo demás, corrijan lo
> que no les cuadre, y completen esas dos antes de entregar. Borren esta nota al terminar.

## 1. Objetivo

Pasar el sistema de las actividades 1 y 2 (predicción de incumplimiento de pago de clientes de
tarjeta de crédito, con un modelo `champion` en MLflow servido por una API) de "entrenado una vez"
a "vigilado en el tiempo": detectar cuándo cambian los datos, alertar cuando el drift supera un
umbral, medir si el modelo sigue funcionando y comprobar los cambios de código con tests
automatizados. Reutiliza el pipeline, el modelo y el champion de la actividad 2 y le agrega la
capa de monitoreo que le faltaba.

## 2. Descripción del trabajo

1. **Drift emulado en la inferencia (`domain/monitoreo.py`, `inferencia_flow.py`).** El lote que
   puntúa cada corrida sale de la partición de **prueba** del dataset (nunca del entrenamiento) y
   se le aplica un drift controlado por `inferencia.nivel_drift` en `configs/config.yaml` (0.0 a
   1.0): desplaza la distribución de una fracción de las variables numéricas crudas y sube la tasa
   de default. Después se recalculan las variables derivadas y se guarda una copia del lote para los
   pipelines de monitoreo.

2. **Tres pipelines de monitoreo en Prefect**, cada uno responde una pregunta distinta:
   - **Drift de features** (`drift_features_flow.py`): ¿cambiaron los datos de entrada? Compara el
     lote contra el entrenamiento con Evidently (reporte HTML por lote; de aquí sale la alerta) y
     con pruebas estadísticas propias por variable -- Kolmogorov-Smirnov (continuas),
     chi-cuadrado (discretas) y PSI (magnitud) -- que explican la alerta y sirven de contraste.
   - **Drift de evento** (`drift_evento_flow.py`): ¿cambió la tasa de default? Chi-cuadrado contra la
     del entrenamiento; rechaza con un mensaje claro los lotes sin etiquetas.
   - **Rendimiento** (`rendimiento_flow.py`): ¿el modelo sigue sirviendo? Calcula el AUC del
     champion sobre el lote (que trae las etiquetas reales) y lo compara con su AUC de holdout;
     registra un run en MLflow con tag `fase=monitoreo`.

3. **Alertas con umbral configurable.** Si la fracción de variables con drift supera
   `monitoreo.umbral_drift` (30%), si el p-valor del evento cae bajo `monitoreo.alfa_evento` (0.05)
   o si el AUC cae más de `monitoreo.tolerancia_auc` (0.05), el pipeline emite una alerta en el log.

4. **Tests (99, sin MLflow ni Prefect).** Datos (esquema, rangos, nulos, categorías permitidas),
   modelo, inferencia, API y lógica de drift. Para poder probar la inferencia sin servicios se sacó
   su lógica a funciones puras (`construir_lote_con_drift`, `puntuar_lote`) y se agregó una
   validación de calidad de datos (`src/datos.py::validar_calidad`).

5. **CI (`.github/workflows/ci.yml`).** GitHub Actions corre `poetry run pytest` en cada pull
   request y en cada push a `master`.

6. **Barrido de niveles (`scripts/barrido_drift.py`).** Mide las tres señales con niveles de drift de
   0.0 a 1.0 contra el champion real, para saber cuánto drift hace falta para que cada alerta salte.

## 3. Decisiones técnicas

| Decisión | Alternativas consideradas | Por qué se eligió |
|---|---|---|
| Aplicar el drift a las columnas **crudas**, antes de la ingeniería de variables | Desplazar también las 8 variables derivadas directamente | Las derivadas se calculan a partir de las crudas; desplazarlas por separado las dejaría inconsistentes entre sí. Con datos reales nadie "inyecta" drift en una variable derivada |
| El lote sale de la partición de **prueba** | Tomarlo de todo el dataset (como en la actividad 2) | Si el lote compartiera filas con el entrenamiento, la comparación de drift no sería justa: la referencia ya las habría visto |
| `tamano_lote = 500` | 100 (el valor heredado de la actividad 2) | Con 100 filas el ruido de muestreo ya marcaba 45% de las variables con drift sin drift real; con 500, 0% (ver sección 6) |
| Una variable tiene drift si el test es significativo **y** el PSI supera 0.10 | Usar solo el p-valor | Con muestras grandes un p-valor diminuto puede salir de un cambio que no importa en la práctica; el PSI aporta el tamaño del efecto |
| La alerta de features sale de Evidently; las pruebas propias son contraste | Alertar con las pruebas propias o con "cualquiera de las dos" | El enunciado pide Evidently para este pipeline; las pruebas propias explican la alerta y avisan si discrepan, sin cambiar el contrato |
| Un pipeline de **rendimiento** aparte | Medir el rendimiento dentro de la inferencia | En producción las etiquetas reales llegan con retraso: medir el rendimiento es un trabajo sobre lotes ya "cerrados", no sobre la predicción del día |
| Comparar contra el AUC de **holdout** del champion | Compararlo contra el AUC de k-fold | El lote sale de la misma partición de prueba que el holdout, así que es la referencia comparable; el k-fold se calculó solo sobre entrenamiento |
| `SUBIDA_TASA_MAXIMA = 0.15` | 0.30, como en el ejemplo de referencia del curso | La tasa real de default es ~22%; +30 puntos la llevaría a más de 50%, poco realista. +15 en el extremo ya produce un drift claro |
| Lógica de drift e inferencia en funciones puras de `domain/` | Dejarla dentro de las tareas de Prefect | El enunciado pide tests sin servicios levantados; mientras viviera dentro de las tareas, no se podía probar |
| Reutilizar el mismo modelo y experimento de MLflow de la actividad 2 | Un modelo separado para la actividad 3 | Es el mismo sistema, ahora con monitoreo encima: se vigila el champion que ya existe |

## 4. Evidencias

- Repositorio: https://github.com/KingAegon19/Actividades-mlops (carpeta `actividad-3/`)
- Pull request de la actividad: https://github.com/KingAegon19/Actividades-mlops/pull/9
- Ejecución del CI en GitHub Actions (check `pruebas` en verde, 99 tests): https://github.com/KingAegon19/Actividades-mlops/actions/runs/43747585688
- Decisiones, hallazgos y tablas completas de cada corrida real: [`PLAN.md`](../PLAN.md)
- Reportes de Evidently de dos corridas reales, abribles en el navegador: [`evidencia/`](./evidencia/) -- sin drift ([`drift_features_nivel_0.0.html`](./evidencia/drift_features_nivel_0.0.html)) y con drift 0.6 ([`drift_features_nivel_0.6.html`](./evidencia/drift_features_nivel_0.6.html))
- Barrido de niveles de drift (gráfico y tabla completa): [`graficos/`](./graficos/)
- [ ] *(Capturas de MLflow y Prefect de las corridas con nivel de drift 0.0, 0.3 y 0.6: runs de
  inferencia, runs con tag `fase=monitoreo` y los flow runs de los pipelines de drift. Ver la tarea 6
  de [`PARA_COMPANEROS.md`](../PARA_COMPANEROS.md).)*

## 5. Resultados

Criterio de éxito, verificado en vivo contra el champion real (`credit-default-modelo` v2,
`random-forest-balanced`), con lotes de 500 clientes:

| `nivel_drift` | Features con drift (Evidently) | Alerta | Tasa de default del lote | Alerta | AUC del lote (holdout: 0.775) | Alerta |
|---|---|---|---|---|---|---|
| 0.0 | 0% | No | 21.0% (entrenamiento: 22.1%) | No | 0.779 | No |
| 0.6 | 61% | **Sí** | 30.0% | **Sí** | 0.678 | **Sí** |

Los reportes de las dos corridas son distintos entre sí, como pide el enunciado. Con nivel 0.6 el
log explica la alerta: las variables que más cambiaron son `PAY_6`, `PAY_4`, `PAY_2`,
`meses_con_atraso` e `interaccion_atraso_x_limite` (estas dos últimas son derivadas: reflejan el
drift de las crudas sin haberlas tocado).

Barrido de niveles de 0.0 a 1.0:

![Barrido de niveles de drift](./graficos/05-barrido-drift.png)

| Nivel | Features con drift (Evidently / pruebas propias) | Tasa de default | AUC del lote | Recall del lote |
|---|---|---|---|---|
| 0.0 | 0% / 0% | 21.0% | 0.779 | 0.648 |
| 0.3 | 39% / 42% | 25.6% | 0.696 | 0.797 |
| 0.6 | 61% / 61% | 30.0% | 0.678 | 0.600 |
| 1.0 | 87% / 90% | 36.0% | 0.516 | 0.906 |

Primer nivel que dispara cada alerta: **features y rendimiento, 0.3; evento, 0.4.**

Lo que muestra:

- **Dos métodos independientes coinciden.** Evidently y las pruebas propias (KS / chi-cuadrado + PSI)
  difieren como máximo 3.2 puntos porcentuales en los 11 niveles y dan el mismo veredicto de alerta
  en todos: que coincidan le da confianza a la alerta.
- **El drift de evento es el más lento en detectarse.** Con 500 filas el error estándar de la tasa
  de default es de ~1.9 puntos; un cambio de +3.5 puntos (nivel 0.3) todavía no es significativo
  (p = 0.07). Las features delatan el problema antes de que el target se mueva.
- **El recall solo habría ocultado la degradación.** Con drift máximo el AUC cae a 0.516 (el modelo ya
  no distingue nada) pero el recall del lote sube a 0.906: con los atrasos desplazados, el modelo
  marca a casi todos como riesgo. Es el caso por el que en la actividad 2 se exigió AUC *y* recall
  para promover un champion, visto ahora desde producción.

Tests: **99 pasan** en local y en el CI de GitHub (51 de la base más 48 nuevos: calidad de datos,
inferencia, pruebas estadísticas y rendimiento).

## 6. Problemas encontrados

- **Falsas alarmas por el tamaño del lote.** Con `tamano_lote = 100` (heredado de la actividad 2) el
  pipeline de drift de features marcaba **45% de las variables con drift incluso sin drift real**
  (`nivel_drift = 0.0`): habría generado una alerta falsa en cada corrida normal. La causa es que
  comparar 100 filas contra ~24,000 de referencia hace que la distancia entre distribuciones salga
  alta solo por el tamaño de muestra, sobre todo en columnas de cola larga (`BILL_AMT*`,
  `PAY_AMT*`). Se midió con `scripts/explorar_tamano_lote.py` (100 filas: 45%; 300: 6%; 500: 0%) y se
  fijó `tamano_lote = 500`. Las pruebas propias dieron 0% con 500, lo que confirma que el problema era
  la muestra y no la herramienta.
- **La inferencia no se podía probar sin servicios.** El enunciado pide tests de la inferencia sin
  MLflow ni Prefect, pero la lógica (armar el lote, aplicar el drift, puntuar) vivía dentro de tareas
  de Prefect. Se refactorizó a funciones puras en `domain/` y se probaron directamente.
- **Docker Desktop no arrancaba del todo** al levantar los servicios para las pruebas en vivo
  (el daemon no respondía aunque el proceso existía). Se resolvió relanzándolo y esperando a que el
  daemon estuviera listo antes de levantar MLflow y Prefect.
- **Limitaciones conocidas** (documentadas en `PLAN.md`):
  - El drift del target se emula convirtiendo 0 en 1 **al azar**, sin relación con las features: es
    ruido de etiqueta puro, que degrada el AUC más de lo que lo haría un cambio de concepto real.
    Sirve para disparar y comparar las alertas, no para estimar la pérdida real de rendimiento.
  - El PSI no tiene tope: con cambios extremos los intervalos quedan vacíos y el valor lo domina el
    piso numérico `EPSILON` (por eso salen valores ~16). Sirve para ordenar variables y compararlo
    contra 0.10 / 0.25, no como número absoluto.
  - El AUC de un lote de 500 filas varía ~±0.03 solo por muestreo; por eso la tolerancia es 0.05.

## 7. Conclusiones

> Esta sección la escriben ustedes, con lo que de verdad aprendieron. Preguntas guía (no tienen que
> contestarlas todas ni en orden):

- [ ] ¿Qué entendieron sobre la diferencia entre **drift de datos**, **drift de evento** y
  **degradación del rendimiento**? ¿Qué les sorprendió de que no se disparen al mismo tiempo (las
  features alertan en 0.3, el target recién en 0.4)?
- [ ] ¿Qué les dice que el recall subiera a 0.91 mientras el AUC caía a 0.52?
- [ ] ¿Qué fue lo más retador de esta actividad y cómo lo resolvieron? (¿el descubrimiento de las
  falsas alarmas por el tamaño del lote? ¿entender qué miden KS, chi-cuadrado y PSI?)
- [ ] ¿Qué papel tiene el CI y por qué importa que los tests no dependan de servicios levantados?
- [ ] Con más tiempo, ¿qué mejorarían? (¿alertas a un archivo o mensaje en vez de solo al log?
  ¿monitoreo programado después de cada inferencia? ¿reentrenar automáticamente cuando hay drift?)

## 8. Uso de herramientas de IA

> Esta sección es un requisito del curso y tiene que reflejar la experiencia real de cada
> quien -- no la completa la IA por ustedes. Preguntas guía:

- [ ] ¿Qué herramienta(s) usaron y para qué tareas concretas de esta actividad?
- [ ] ¿Qué decidieron o hicieron ustedes sin IA? (qué señales de monitoreo construir, interpretar
  los resultados, qué niveles de drift probar, qué tareas repartir al equipo...)
- [ ] ¿Qué propuso la IA que ustedes revisaron, corrigieron o descartaron, y por qué? (por ejemplo,
  qué pruebas estadísticas usar para detectar el drift, o qué tanta profundidad darle a la actividad)
- [ ] ¿Qué podrían explicar hoy sin la IA, de punta a punta, sobre cómo funciona el monitoreo?
