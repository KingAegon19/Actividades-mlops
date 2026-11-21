# Para el equipo: cómo contribuir código a actividad-3

Guía para los otros dos integrantes (o su IA, si usan una). Con esto alcanza para tomar una tarea
sin romper nada de lo que ya funciona. El estado completo y las decisiones están en
[`PLAN.md`](./PLAN.md); cómo correr todo, en [`README.md`](./README.md).

## Estado actual

> **Actualización:** la tarea 3 (falsas alarmas) la hizo José y la memoria (tarea 7) la completó Emma;
> ambas están integradas. Las tareas 1, 2, 4, 5 y 6 siguen abiertas como mejoras opcionales.

Actividad 3 está **completa a nivel de código y funcionando**: drift emulado en la inferencia, tres
pipelines de monitoreo (features, evento, rendimiento), pruebas estadísticas por variable, validación
de calidad de datos, 99 tests y CI en verde. Lo que falta es **pulir y ampliar**, y la Memoria 3.

## Reglas (aprendidas en actividad 2 -- léelas, son las que sí se rompieron)

1. **Rama propia, y el PR apunta a `feature/actividad-3-scaffold`, NO a `master`.** `master` todavía
   no tiene actividad-3 (se mergea al final, cuando el grupo decida). En actividad 2 un PR salió
   apuntando a `master` por error y habría publicado todo antes de tiempo. Al abrir el PR, revisa
   que el campo "base" diga `feature/actividad-3-scaffold`.
   ```powershell
   git fetch origin
   git checkout -b feature/<lo-que-hagas> origin/feature/actividad-3-scaffold
   ```
2. **Antes de pushear, corre los tests** y que pasen todos:
   ```powershell
   cd actividad-3
   poetry install
   poetry run pytest -q
   ```
   El CI de GitHub corre lo mismo en cada PR: si falla, no se mergea.
3. **Si tu tarea necesita el champion, pruébala contra el MLflow compartido del equipo**, no contra
   uno local vacío. En actividad 2, un modelo "ganó" el champion en el MLflow local de un
   integrante porque no tenía contra quién competir -- y no era cierto en el estado real del proyecto.
4. **No copies código de `../../actividades/`** (el repo del profesor). Puedes leerlo para entender
   un patrón, pero lo que escribas tiene que ser propio y adaptado a nuestro caso.
5. **No cambies las funciones existentes de `domain/` ni los umbrales de `configs/config.yaml`**
   sin avisar: los tests y los resultados documentados en `PLAN.md` dependen de ellos. Agregar es
   seguro; modificar, consúltalo.

## Tareas de código (de menos a más técnicas)

Cada una indica qué archivos toca, para que dos personas no pisen los mismos.

### 1. Tests de borde (riesgo bajo, solo agrega archivos de test)
**Toca:** `tests/` (archivos nuevos o ampliar los existentes)

Casos que hoy no están cubiertos, por ejemplo: `aplicar_drift` con una columna constante o con un
lote de pocas filas; `drift_de_target` con un lote de una sola clase; `validar_calidad` con varias
reglas rotas a la vez (¿devuelve todos los problemas?); `psi_continua` con valores negativos. Si un
test falla porque encontraste un bug real, eso es un hallazgo valioso: documéntalo (va directo a la
sección "Problemas encontrados" de la Memoria 3).

### 2. Gráfico de variables con más drift (riesgo bajo, archivo nuevo)
**Toca:** `scripts/grafico_psi.py` (nuevo), `docs/graficos/README.md`

Lee `reports/drift_features_lote_actual_variables.csv` (lo genera `drift_features_flow.py`) y dibuja
las 10 variables con mayor PSI como barras horizontales, marcando los umbrales 0.10 y 0.25. Usa la
misma paleta y estilo que `scripts/barrido_drift.py` para que se vea consistente.

### 3. Tasa de falsas alarmas con muchas semillas (riesgo bajo, archivo nuevo)
**Toca:** `scripts/falsas_alarmas.py` (nuevo), `PLAN.md` (agregar una sección con el resultado)

Hoy sabemos que con `tamano_lote=500` y la semilla 42 no hay alarma falsa en nivel 0.0. ¿Y con otras
50 semillas? Repite el armado del lote cambiando `division["random_state"]` (recalcula también la
referencia con la misma división) y mide: (a) % de corridas con alarma en nivel 0.0 (falsas alarmas)
y (b) % de corridas con alarma en nivel 0.3 (poder de detección). Es la prueba de robustez que falta
para confiar en el umbral del 30%. No necesita MLflow.

### 4. Desplegar los pipelines de monitoreo en Prefect (riesgo medio)
**Toca:** `infraestructure/flows/desplegar.py`

Hoy solo se despliegan entrenamiento (manual) e inferencia (diaria). Agrega las tres de monitoreo
(`drift_features_flow`, `drift_evento_flow`, `rendimiento_flow`) con `.to_deployment(...)`: pueden
ser manuales o con un horario posterior a la inferencia (ej. 06:30 UTC). Verifica en la UI de Prefect
(`http://localhost:4200`) que aparezcan como despliegues y que una ejecución manual funcione.

### 5. Alertas a un archivo, no solo al log (riesgo medio)
**Toca:** `domain/alertas.py` (nuevo), las tareas `alerta` de los tres flows de monitoreo, `tests/`

Hoy una alerta es un `print` en el log de Prefect. Crea `emitir_alerta(tipo, mensaje, ruta)` que además
agregue una línea JSON (fecha, tipo, mensaje) a `reports/alertas.jsonl`, llámala desde los tres flows
y pruébala con `tmp_path`. El enunciado deja el medio de la alerta a nuestra elección ("un log o un
mensaje"), así que es una mejora real. **Una sola persona** debería tomar esta, porque edita los tres flows.

### 6. Evidencia en vivo (sin código nuevo, pero necesaria para la Memoria 3)
**Toca:** `docs/evidencia/` (capturas nuevas)

Con los servicios levantados, corre inferencia + los tres pipelines de monitoreo con
`inferencia.nivel_drift` en 0.0, 0.3 y 0.6 (cambiándolo en `configs/config.yaml`, **y restaurándolo a
0.0 antes de commitear**). Captura: la lista de runs en MLflow (inferencia y `fase=monitoreo`), los
flow runs en Prefect, y la alerta en el log. El nivel 0.3 todavía no tiene capturas.

### 7. Memoria 3 (sin tocar código, pero es lo único que falta del entregable)
**Toca:** `docs/memoria-3.md`

Las secciones 1-6 ya están armadas con lo técnico. **Faltan la 7 (Conclusiones) y la 8 (Uso de IA)**,
con preguntas guía. Escríbanlas con su propia voz y su propia experiencia (igual que en la Memoria 2,
en voz de equipo: "aprendimos...", "usamos..."): son justo las secciones donde se nota el aporte de
cada quien, y la sección 8 tiene que ser honesta sobre qué hicieron con IA y qué sin ella. Si hacen la
tarea 6 (capturas), pongan los enlaces en la sección 4, donde hay un `[ ]` esperándolos.

## Dudas

Si algo no cuadra, pregunta en el grupo antes de pushear: es más fácil resolver una duda que deshacer un PR.
