import pandas as pd

from src.config import RAIZ_PROYECTO

# Ver actividad-1/docs/diccionario-datos.md para el detalle de cada columna.
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


def cargar_datos(ruta: str, objetivo: str) -> pd.DataFrame:
    """Carga el CSV, valida sus columnas y descarta el identificador (no es feature)."""
    ruta_completa = RAIZ_PROYECTO / ruta
    df = pd.read_csv(ruta_completa)

    columnas_faltantes = set(COLUMNAS_ESPERADAS) - set(df.columns)
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {sorted(columnas_faltantes)}")

    if objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{objetivo}' no está en el dataset")

    return df.drop(columns=["ID"])


def agrupar_categorias_sin_documentar(df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa los valores de EDUCATION/MARRIAGE sin categoría oficial en 'otros'."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace(EDUCATION_SIN_DOCUMENTAR)
    df["MARRIAGE"] = df["MARRIAGE"].replace(MARRIAGE_SIN_DOCUMENTAR)
    return df
