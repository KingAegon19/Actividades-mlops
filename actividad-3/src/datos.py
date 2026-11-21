import pandas as pd

from src.config import RAIZ_PROYECTO

# Ver ../actividad-1/docs/diccionario-datos.md para el detalle de cada columna.
COLUMNAS_ESPERADAS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
    "default.payment.next.month",
]

# EDUCATION: 5, 6 y 0 no tienen categoría documentada por los autores originales -> se agrupan con "otros" (4).
# MARRIAGE: 0 no tiene categoría documentada -> se agrupa con "otros" (3).
EDUCATION_SIN_DOCUMENTAR = {0: 4, 5: 4, 6: 4}
MARRIAGE_SIN_DOCUMENTAR = {0: 3}


# Rangos y valores permitidos, comprobados contra el dataset real (ver PLAN.md, "tests de datos").
# Son deliberadamente algo más anchos que lo observado (ej. AGE real 21-79, aquí 18-100) para
# detectar datos corruptos o fuera de dominio, no para rechazar clientes legítimos poco comunes.
RANGOS_NUMERICOS = {
    "LIMIT_BAL": (1, None),  # el límite de crédito debe ser positivo
    "AGE": (18, 100),
    **{f"PAY_{i}": (-2, 9) for i in ("0", "2", "3", "4", "5", "6")},  # -2 = sin consumo ... 9 = 9+ meses de atraso
    **{f"PAY_AMT{i}": (0, None) for i in range(1, 7)},  # un pago no puede ser negativo
}
VALORES_PERMITIDOS = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},  # ya agrupadas las categorías sin documentar
    "MARRIAGE": {1, 2, 3},
}


def validar_calidad(df: pd.DataFrame, objetivo: str) -> list[str]:
    """Revisa nulos, rangos y valores permitidos. Devuelve la lista de problemas (vacía si el
    dataset es válido) -- no lanza nada, para poder probarla y para que quien la use decida."""
    problemas = []

    nulos = df.isna().sum()
    for columna, cantidad in nulos[nulos > 0].items():
        problemas.append(f"'{columna}' tiene {cantidad} valores nulos")

    for columna, (minimo, maximo) in RANGOS_NUMERICOS.items():
        if columna not in df.columns:
            continue
        if minimo is not None and (df[columna] < minimo).any():
            problemas.append(f"'{columna}' tiene valores por debajo de {minimo} (mínimo observado: {df[columna].min()})")
        if maximo is not None and (df[columna] > maximo).any():
            problemas.append(f"'{columna}' tiene valores por encima de {maximo} (máximo observado: {df[columna].max()})")

    for columna, permitidos in VALORES_PERMITIDOS.items():
        if columna not in df.columns:
            continue
        inesperados = set(df[columna].dropna().unique()) - permitidos
        if inesperados:
            problemas.append(f"'{columna}' tiene valores no permitidos: {sorted(inesperados)}")

    if objetivo in df.columns:
        inesperados = set(df[objetivo].dropna().unique()) - {0, 1}
        if inesperados:
            problemas.append(f"'{objetivo}' debe ser 0 o 1, pero tiene: {sorted(inesperados)}")

    return problemas


def cargar_datos(ruta: str, objetivo: str) -> pd.DataFrame:
    """Carga el CSV, valida sus columnas, descarta el identificador, agrupa categorías sin
    documentar y valida la calidad (nulos, rangos) -- falla ruidosamente si algo no cuadra."""
    ruta_completa = RAIZ_PROYECTO / ruta
    df = pd.read_csv(ruta_completa)

    columnas_faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {sorted(columnas_faltantes)}")

    if objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{objetivo}' no está en el dataset")

    df = agrupar_categorias_sin_documentar(df.drop(columns=["ID"]))

    problemas = validar_calidad(df, objetivo)
    if problemas:
        raise ValueError("El dataset no pasa la validación de calidad: " + "; ".join(problemas))
    return df


def agrupar_categorias_sin_documentar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa los valores de EDUCATION/MARRIAGE sin categoría oficial en 'otros'."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace(EDUCATION_SIN_DOCUMENTAR)
    df["MARRIAGE"] = df["MARRIAGE"].replace(MARRIAGE_SIN_DOCUMENTAR)
    return df
