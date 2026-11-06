# Para el equipo: cómo contribuir a actividad-2

Esta guía es para los otros dos integrantes del grupo (o su IA, si usan una para ayudarles).
No necesitan leer todo el historial del proyecto -- con esto alcanza para entender el estado
actual y tomar una tarea sin romper nada de lo que ya funciona.

## Estado actual (resumen)

Actividad 2 ya está **completa a nivel de código**: pipeline de entrenamiento e inferencia en
Prefect, conectados por MLflow (alias `champion`), API con FastAPI, todo configurado en YAML,
39 tests pasando. El detalle completo de decisiones y resultados está en
[`PLAN.md`](./PLAN.md); cómo correr todo, en [`README.md`](./README.md).

**Lo único que falta del entregable es la Memoria 2** (el documento escrito). Por eso la
mayoría de tareas sugeridas abajo son de documentación/pruebas, no de reescribir el pipeline.

## Reglas antes de empezar

1. **Nunca pusheen directo a `feature/actividad-2-scaffold`.** Cada quien trabaja en su propia
   rama (`feature/<lo-que-sea-que-hagan>`) y abre un Pull Request contra
   `feature/actividad-2-scaffold`. Esa rama en sí todavía no se mergea a `main` -- queda así
   a propósito hasta que el grupo decida que está lista.
2. **Antes de pushear, corran los tests:**
   ```powershell
   cd actividad-2
   poetry install
   poetry run pytest -q
   ```
   Si algo falla, no pusheen -- arreglen primero o pregunten.
3. **No copien código de `../actividades/`** (el repo de referencia del profesor). Pueden
   mirarlo para entender un patrón (ej. cómo se usa `mlflow.sklearn.load_model`), pero el
   código que escriban tiene que ser propio, adaptado a nuestro caso (default de tarjeta de
   crédito), no una copia.
4. Lean [`AGENTS.md`](../AGENTS.md) en la raíz del repo si usan un asistente de IA -- tiene
   las reglas del proyecto (estructura, convenciones, qué no copiar).

## Tareas sugeridas (de menos a más técnicas)

### 1. Memoria 2 (sin tocar código)

El documento que falta. Usar el mismo formato que la Memoria 1 (ver
`../actividades/docs/plantilla-memoria.md` si existe, o el formato que ya usamos). El
material ya está casi todo armado:
- `PLAN.md` tiene las decisiones y resultados reales de cada corrida.
- `docs/graficos/` tiene 4 gráficos explicativos (AUC, ROC, k-fold).
- `docs/evidencia/` tiene 8 capturas reales del proceso (Model Registry, Prefect, API).

### 2. Documentación (bajo riesgo)

Mejorar `README.md`, `PLAN.md`, o agregar comentarios donde algo no quede claro. No cambia
comportamiento de nada.

### 3. Tests nuevos (riesgo bajo, aditivo)

Agregar casos a `tests/` que no existan todavía -- por ejemplo, más validaciones en
`tests/test_datos.py` (¿qué pasa si el CSV tiene una columna de más?) o en
`tests/test_ingenieria.py`. Los 39 tests actuales sirven de red de seguridad: si algo se
rompe con un cambio, los tests existentes lo muestran antes de pushear.

### 4. Un candidato de modelo nuevo (riesgo medio, pero aislado)

Agregar otro algoritmo al factory de modelos, siguiendo el mismo patrón que ya existe para
`logistic_regression`, `random_forest` y `xgboost`:

- En `domain/modelo.py`, función `construir_modelo`: agregar un `elif tipo == "nombre_nuevo"`.
- En `configs/config.yaml`, agregar un candidato nuevo bajo `candidatos:` con ese `tipo`.
- Correr `entrenamiento_flow.py` (servicios de MLflow/Prefect deben estar levantados, ver
  `README.md`) y ver si gana contra el champion actual.

Esto toca solo dos archivos y no afecta nada de lo que ya funciona, salvo que el modelo nuevo
gane (lo cual está bien -- es justo la idea).

## Dudas

Si algo no cuadra, pregunten en el grupo antes de pushear algo que no están seguros si rompe
el pipeline -- es más fácil resolver una duda antes que deshacer un PR.
