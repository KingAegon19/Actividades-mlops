# Diccionario de datos: Default of Credit Card Clients

**Fuente:** [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI, Taiwan). Clientes de tarjeta de crédito, abril-septiembre 2005.
**Archivo:** `data/credit_card_default.csv`
**Dimensiones:** 30,000 filas × 25 columnas. Sin nulos, sin duplicados en `ID`.
**Objetivo:** predecir si un cliente **incumplirá su pago el mes siguiente** (octubre 2005).

## Variable objetivo (target)

| Columna | Significado | Valores | Distribución real |
|---|---|---|---|
| `default.payment.next.month` | 1 = el cliente incumplió el pago del mes siguiente; 0 = pagó a tiempo | `{0, 1}` | 77.9 % no incumple (0) / 22.1 % incumple (1) — **desbalanceado**, considerar en la evaluación (no usar solo accuracy) |

## Identificador (no es feature)

| Columna | Significado |
|---|---|
| `ID` | Identificador único del cliente. Se descarta antes de entrenar: no aporta señal predictiva, solo sirve para trazabilidad de filas. |

## Features demográficas

| Columna | Significado | Valores | Notas |
|---|---|---|---|
| `LIMIT_BAL` | Monto del crédito otorgado (NT dollar), incluye crédito individual y familiar/suplementario | numérico continuo, 10,000 – 1,000,000 | sin nulos |
| `SEX` | Género | `1` = masculino, `2` = femenino | — |
| `EDUCATION` | Nivel educativo | `1` = posgrado, `2` = universidad, `3` = bachillerato, `4` = otros, `5`/`6`/`0` = sin documentar en la fuente original | 14 + 280 + 51 filas (`0`,`5`,`6`) no tienen categoría oficial: decidir si se agrupan en "otros" o se excluyen |
| `MARRIAGE` | Estado civil | `1` = casado, `2` = soltero, `3` = otros, `0` = sin documentar | 54 filas con `0` |
| `AGE` | Edad en años | numérico, 21 – 79 | — |

## Features de historial de pago (los 6 meses previos, abril–septiembre 2005)

| Columnas | Significado | Valores |
|---|---|---|
| `PAY_0` (= septiembre), `PAY_2` (agosto), `PAY_3` (julio), `PAY_4` (junio), `PAY_5` (mayo), `PAY_6` (abril) | Estado de pago de ese mes | `-2`/`-1`/`0` = sin atraso (significado exacto de `-2`/`0` no está documentado por los autores originales, pero empíricamente indican "sin consumo" o "pagado dentro del mínimo"); `1`-`8` = meses de atraso (1 = un mes de atraso, 8 = ocho meses o más) |

**Nota importante:** la columna del mes de septiembre se llama `PAY_0`, no `PAY_1` — es una inconsistencia de nombres de los autores originales del dataset, hay que tenerla presente al documentar el código.

## Features de montos (los 6 meses previos)

| Columnas | Significado | Valores |
|---|---|---|
| `BILL_AMT1` (septiembre) … `BILL_AMT6` (abril) | Monto del estado de cuenta (factura) de ese mes, en NT dollar | numérico, puede ser negativo (saldo a favor del cliente) |
| `PAY_AMT1` (septiembre) … `PAY_AMT6` (abril) | Monto efectivamente pagado ese mes, en NT dollar | numérico, ≥ 0 |

## Resumen para el modelado

- **Features:** las 23 columnas anteriores (demográficas + historial de pago + montos), sin `ID`.
- **Target:** `default.payment.next.month`.
- **Partición:** usar `stratify` por el target al hacer train/test split, porque está desbalanceado (22 % positivos).
- **Variables categóricas** (`SEX`, `EDUCATION`, `MARRIAGE`) ya vienen codificadas como enteros en el CSV — decidir si se tratan como categóricas (one-hot) o se dejan ordinales según el modelo que se use.
- **Valores sin documentar** (`EDUCATION` en `{0,5,6}`, `MARRIAGE` en `{0}`) son pocas filas (<1.5 % del total): se puede agrupar en una categoría "otros/desconocido" en vez de eliminarlas.

Este diccionario es la base para `src/datos.py` (carga y validación de columnas) y para el flow de la actividad 1.
