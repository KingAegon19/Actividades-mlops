from pathlib import Path

import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent


def cargar_config(ruta: str = "configs/config.yaml") -> dict:
    """Carga el YAML de configuración, resolviendo la ruta desde la raíz del proyecto."""
    ruta_completa = RAIZ_PROYECTO / ruta
    with open(ruta_completa, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
