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
| `configs/features.yaml` | `ALL` = 23 columnas; `NUMERIC` = todo menos `SEX`/`EDUCATION`/`MARRIAGE`; `CUSTOM` = **confirmado por EDA**: límite de crédito + historial de pago (`PAY_0`..`PAY_6`) -- son las 7 variables con mayor correlación real con el target (ver `notebooks/eda_credit_default.ipynb`) |
| Lote de "clientes nuevos" para inferencia | Pendiente de decidir: ¿se aparta un % del dataset desde el inicio (ej. 10%) que entrenamiento nunca toca? |
| Champion | `registro.umbral_auc` en `config.yaml`, provisional en 0.70 (el baseline de actividad 1 dio AUC ≈ 0.71) |

## Decisiones del EDA (`notebooks/eda_credit_default.ipynb`)

- **Variables más predictivas:** el historial de pago domina (`PAY_0` r=0.32, bajando gradualmente hasta `PAY_6` r=0.19), seguido de `LIMIT_BAL` (r=-0.15). `BILL_AMT*` son casi irrelevantes solas (|r| < 0.02).
- **`CUSTOM` confirmado:** `LIMIT_BAL` + `PAY_0`..`PAY_6` (coincide con las variables de mayor correlación real).
- **Escalar features: sí, hace falta.** Los montos llegan a cientos de miles mientras el resto de variables va de -2 a 8 -- el Modelo 1 (logistic regression) debe llevar `StandardScaler`.
- **Categorías sin documentar** (`EDUCATION` en 0/5/6, `MARRIAGE` en 0) tienen muy pocas filas y tasas erráticas -- confirma que agruparlas en "otros" (como ya hace `src/datos.py` de actividad-1) es la decisión correcta.

## Decisiones pendientes (antes de escribir los flows)

- [ ] Confirmar el umbral de AUC para promoción a champion (0.70 es un punto de partida, no una decisión final).
- [ ] Definir cómo se separa el lote de inferencia del de entrenamiento (holdout fijo vs. muestra aleatoria del test set).
- [ ] Decidir si `actividad-2/` reusa el dataset de `actividad-1/data/` (como está configurado ahora) o tiene su propia copia.

## Estructura creada (sin lógica todavía)

```
actividad-2/
├── configs/
│   ├── config.yaml        # datos, modelo, MLflow, umbral de champion, inferencia
│   └── features.yaml      # selección de variables (ALL/NUMERIC/CUSTOM)
├── domain/                # lógica de negocio (pendiente)
├── infraestructure/
│   ├── flows/             # entrenamiento_flow.py, inferencia_flow.py, desplegar.py (pendientes)
│   └── api/                # main.py de FastAPI (pendiente)
├── src/                   # utilidades compartidas (config, datos) -- igual patrón que actividad-1
├── data/                  # salida de predicciones de inferencia (gitignored)
├── tests/
└── pyproject.toml         # Poetry: prefect, mlflow, scikit-learn, pandas, numpy, pyyaml, fastapi, uvicorn
```
