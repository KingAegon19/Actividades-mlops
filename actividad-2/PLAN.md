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
| `configs/features.yaml` | `ALL` = 23 columnas; `NUMERIC` = todo menos `SEX`/`EDUCATION`/`MARRIAGE`; `CUSTOM` = provisional: límite de crédito + historial de pago (`PAY_0`..`PAY_6`), sin datos demográficos |
| Lote de "clientes nuevos" para inferencia | Pendiente de decidir: ¿se aparta un % del dataset desde el inicio (ej. 10%) que entrenamiento nunca toca? |
| Champion | `registro.umbral_auc` en `config.yaml`, provisional en 0.70 (el baseline de actividad 1 dio AUC ≈ 0.71) |

## Decisiones pendientes (antes de escribir los flows)

- [ ] Confirmar el umbral de AUC para promoción a champion (0.70 es un punto de partida, no una decisión final).
- [ ] Definir cómo se separa el lote de inferencia del de entrenamiento (holdout fijo vs. muestra aleatoria del test set).
- [ ] Confirmar el conjunto `CUSTOM` de `features.yaml` (¿tiene sentido para un caso de riesgo de crédito excluir variables demográficas?).
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
