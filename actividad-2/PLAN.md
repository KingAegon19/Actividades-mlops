# Plan de actividad 2 (borrador, antes de implementar)

Referencia: `../../actividades/docs/actividad-2.md` (guía del profesor). No implementar la lógica todavía — esto es solo el plan y la estructura de carpetas, a completar cuando actividad 1 esté cerrada.

## Qué pide la actividad

1. **Pipeline de entrenamiento** en Prefect: ingesta → entrenar → evaluar → registrar. Cada run con tag `fase=entrenamiento`.
2. El modelo nuevo se compara con el **champion** actual (alias en el MLflow Model Registry, no número de versión). Solo reemplaza al champion si supera un AUC mínimo **y** el AUC del champion vigente.
3. **Pipeline de inferencia** en Prefect: carga el modelo por el alias `champion`, puntúa un lote de "clientes nuevos" simulados, guarda las predicciones con la fecha de ejecución. Desplegada con horario **diario**.
4. **API con FastAPI** que sirve el `champion` y expone un endpoint de predicción, con un ejemplo documentado.
5. Configuración en YAML (`configs/config.yaml`, `configs/features.yaml`), nunca hardcodeada.

## Cómo se traduce a nuestro caso de uso

| Pieza | En nuestro proyecto (default de tarjeta de crédito) |
|---|---|
| `domain/modelo.py` | Lógica pura (pipeline sklearn + selección de features por alias), sin Prefect/MLflow |
| Carga de datos | Reutiliza `src/datos.py` de actividad-1, o se duplica aquí si conviene mantener cada actividad autocontenida |
| `configs/features.yaml` | `seleccion: ALL\|NUMERIC\|CUSTOM` (como pide literalmente el enunciado). `CUSTOM` **confirmado por EDA**: límite de crédito + historial de pago `PAY_0`..`PAY_6`, las 7 variables con mayor correlación real con el target -- ver `notebooks/eda_credit_default.ipynb` |
| Lote de "clientes nuevos" para inferencia | **Resuelto**: muestra aleatoria de `inferencia.tamano_lote` filas tomada del CSV completo en cada ejecución (simulación, no un holdout real) — cambia cada vez que corre, documentado como tal |
| Champion | `registro.umbral_auc` en `config.yaml`, provisional en 0.70 (el baseline de actividad 1 dio AUC ≈ 0.71) |

## El mecanismo de promoción a champion (MLflow Model Registry)

Usa 3 llamadas de la API de MLflow, sin lógica propia que reinventar:

```python
mlflow.register_model(model_uri, nombre)                           # crea una version nueva del modelo
client.get_model_version_by_alias(nombre, "champion")               # None si todavia no existe ninguno
client.set_registered_model_alias(nombre, "champion", version)      # mueve el alias a la version nueva
```

Regla: solo se promueve si `auc_nuevo >= registro.umbral_auc` **y** `recall_nuevo >= registro.umbral_recall` **y** (no hay champion todavía, o `auc_nuevo > auc_champion_actual`). Si no, el run queda registrado en MLflow (consultable) pero el alias no se mueve.

**Por qué el recall mínimo:** el AUC solo dice qué modelo ordena mejor el riesgo en general, pero no depende de ningún umbral de decisión -- un modelo con AUC alto igual podría tener mal recall en la práctica. Y el recall solo no sirve como único criterio: un modelo que marque a *todos* los clientes como riesgo tiene recall = 1.0 perfecto y es inútil (el AUC lo filtra, porque también penaliza marcar como riesgo a quien no lo es). Por eso se exigen los dos: AUC para "¿el modelo es bueno en general?" y recall mínimo para "¿de verdad atrapa los defaults que nos importan?". Dado nuestro caso (perder un default real pesa más que una falsa alarma), un modelo no debería ser champion solo por tener buen AUC si su recall es pobre.

## Decisiones del EDA (`notebooks/eda_credit_default.ipynb`)

- **Variables más predictivas:** el historial de pago domina (`PAY_0` r=0.32, bajando gradualmente hasta `PAY_6` r=0.19), seguido de `LIMIT_BAL` (r=-0.15). `BILL_AMT*` son casi irrelevantes solas (|r| < 0.02).
- **`CUSTOM` confirmado:** `LIMIT_BAL` + `PAY_0`..`PAY_6` (coincide con las variables de mayor correlación real).
- **Escalar features: sí, hace falta.** Los montos llegan a cientos de miles mientras el resto de variables va de -2 a 8 -- el Modelo 1 (logistic regression) debe llevar `StandardScaler`.
- **Categorías sin documentar** (`EDUCATION` en 0/5/6, `MARRIAGE` en 0) tienen muy pocas filas y tasas erráticas -- confirma que agruparlas en "otros" (como ya hace `src/datos.py` de actividad-1) es la decisión correcta.

## Horario de los pipelines (Prefect deployments)

**Lo obligatorio según el enunciado literal:** la inferencia debe tener horario diario; el entrenamiento *"puede lanzarse a mano"* -- no exige ningún horario. Eso es lo que hay que tener cerrado primero.

**Extra opcional (como el paso 4 de actividad 1):** el profesor comentó en clase la idea de que el entrenamiento no corra una sola vez, sino que se agende para que, cada mes, al llegar datos nuevos, reentrene el paquete de candidatos (`candidatos` en `config.yaml`) y revalide el champion. No es un entregable del enunciado escrito, pero no contradice nada y profundiza el ejercicio -- se agrega después de que el despliegue manual + diario ya funcione.

| Pipeline | Horario (cron, UTC) | Obligatorio? |
|---|---|---|
| Entrenamiento | Manual (UI de Prefect) -- opcionalmente también mensual (`0 6 1 * *`) | No; el mensual es el extra |
| Inferencia | Diario (`0 6 * * *`) | Sí |

## Primera corrida real de entrenamiento (`entrenamiento_flow.py`)

| Candidato | AUC test | AUC train | dif (overfitting) | Recall |
|---|---|---|---|---|
| logistic-regression | 0.7100 | 0.7296 | +0.02 | 0.2442 |
| random-forest (sin `max_depth`) | 0.7595 | **1.0000** | **+0.24 -- overfitting real** | 0.3730 |
| random-forest (`max_depth: 8`) | 0.7742 | 0.8187 | +0.04 | 0.3451 |
| logistic-regression-balanced (`class_weight: balanced`) | 0.7104 | 0.7301 | +0.02 | 0.6307 |
| **random-forest-balanced** (`max_depth: 8`, `class_weight: balanced`) | **0.7743** | 0.8248 | +0.05 | **0.5998** |

Hallazgos:
- **El random forest sin límite de profundidad memorizaba el train** (AUC train = 1.0000 exacto): no era un modelo bueno, era overfitting disfrazado de buen AUC de test. Se corrigió agregando `max_depth: 8` -- la diferencia train-test bajó de +0.24 a +0.04/+0.05, un nivel sano.
- **`class_weight: balanced` fue una mejora real del modelo** (no solo mover el umbral de decisión): subió el recall de la regresión logística de 0.24 a 0.63, y el del random forest de 0.35 a 0.60, con el AUC prácticamente igual o mejor.
- `random-forest-balanced` ganó la tanda (mejor AUC) y cumplió ambos umbrales de champion (AUC 0.7743 ≥ 0.70, recall 0.5998 ≥ 0.50) → quedó registrado como la **versión 1, primer champion real** en el Model Registry.

## Primera corrida real de inferencia (`inferencia_flow.py`)

Cargó el champion por alias (`credit-default-modelo` versión 1, sin número hardcodeado), tomó
una muestra de 100 clientes simulados del CSV, calculó la probabilidad de default
(media = 0.4330) y guardó `data/prediccion_2026-11-05.csv`. El run de inferencia en MLflow
quedó con tag `fase=inferencia` y conectado al `run_id` de entrenamiento que generó ese
champion -- se puede rastrear cualquier predicción hasta el modelo exacto que la hizo.

## Decisiones pendientes (antes de escribir los flows restantes)

- [x] Confirmar el umbral de AUC para promoción a champion: 0.70 se valida con la corrida real (el champion quedó en 0.7743).
- [x] Cómo se simula el lote de inferencia: muestra aleatoria del CSV completo (ver tabla arriba).
- [x] Horario de los pipelines: inferencia diaria es obligatoria; entrenamiento manual obligatorio, mensual es extra opcional (ver tabla arriba).
- [x] `actividad-2/` tiene su propia copia del dataset en `data/credit_card_default.csv` (autocontenida, no depende de `actividad-1/`).

## Nota de entorno: `PYTHONUTF8`

Ya nos topamos en actividad 1 con un `UnicodeEncodeError` al imprimir el emoji que MLflow muestra al cerrar un run, en consolas con codificación `cp1252` (Git Bash en Windows). La forma robusta de evitarlo, en vez de depender de usar PowerShell: definir `$env:PYTHONUTF8 = "1"` junto a las otras variables de entorno antes de correr cualquier flow.

## Estructura creada (sin lógica todavía)

```
actividad-2/
├── configs/
│   ├── config.yaml        # datos, candidatos, MLflow, umbral de champion/recall, inferencia
│   └── features.yaml      # seleccion: ALL/NUMERIC/CUSTOM (como pide el enunciado literal)
├── domain/                # lógica de negocio (pendiente)
├── infraestructure/
│   ├── flows/             # entrenamiento_flow.py, inferencia_flow.py, desplegar.py (pendientes)
│   └── api/                # main.py de FastAPI (pendiente)
├── src/                   # utilidades compartidas (config, datos) -- igual patrón que actividad-1
├── data/                  # salida de predicciones de inferencia (gitignored)
├── tests/
└── pyproject.toml         # Poetry: prefect, mlflow, scikit-learn, pandas, numpy, pyyaml, fastapi, uvicorn
```
