# AGENTS.md — contexto para agentes de IA

Guía para cualquier asistente de IA (Claude Code, Gemini/Antigravity, Cursor, etc.) que trabaje en este repositorio. Léela completa antes de proponer o escribir código.

## Qué es este repositorio

Repositorio **personal** de un estudiante para la materia universitaria de MLOps (curso "MLOps con Prefect y MLflow"). Aquí se desarrollan las actividades 1-4 de la materia con un caso de uso propio del estudiante.

**No confundir con el repositorio del profesor.** Al lado de este repo, en `../actividades/`, está clonado por separado el contenido que el profesor comparte (`github.com/henrycala/santoto-mlops-content`): ahí están la **solución de referencia** de las actividades 1-3, la documentación/rúbrica de cada actividad (`docs/actividad-1.md` … `actividad-4.md`), la plantilla de memoria (`docs/plantilla-memoria.md`) y el calendario.

**Regla de integridad académica — la más importante de este archivo:** nunca copiar código de `../actividades/` hacia este repo. Esa carpeta es solo para consultar la rúbrica y entender el patrón esperado (estructura de carpetas, cómo se usa MLflow/Prefect). El caso de uso, los datos y el código de este repo deben ser propios. Sí es correcto reutilizar la infraestructura genérica (el `services/docker-compose.yml`, idéntico al del curso) porque no es "la solución", es solo el levantamiento de MLflow/Prefect.

## Quién soy (el estudiante)

Analista de bases de datos en el área de **contaduría pública**. Nuevo en Docker, Poetry y el stack de MLOps — las explicaciones deben partir de fundamentos, no asumir experiencia previa con estas herramientas. Cuenta de GitHub: `KingAegon19`.

Tengo un proyecto de tesis separado que **no** se mezcla con el trabajo de esta materia.

## Caso de uso elegido (actividad 1-3)

**Predicción de incumplimiento de pago (default) de clientes de tarjeta de crédito** a partir de su historial de pagos, límite de crédito y estados de cuenta — elegido por su relación con riesgo de crédito/cuentas por cobrar en contaduría pública, y por tener suficiente volumen (30,000 filas) para que el ejercicio de MLOps se sienta representativo.

- Dataset: [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI, Taiwan). 30,000 clientes, 23 variables (límite de crédito, historial de pagos, montos de factura/pago, demográficos) + columna objetivo `default.payment.next.month`.
- Va en `actividad-1/data/credit_card_default.csv` (no versionado en git, se descarga con la Kaggle CLI autenticada vía `kaggle auth login`).
- Es clasificación binaria, igual que el ejemplo de churn del profesor — se puede seguir ese mismo patrón de pipeline (cargar → entrenar → trackear) sin copiar su código.

## Stack técnico

| Pieza | Herramienta | Notas |
|---|---|---|
| Orquestación | **Prefect** 3.8.8 | Tope de Python <3.15 por esta versión |
| Tracking/registro | **MLflow** 3.17.0 | UI en `localhost:5000` |
| Dependencias por actividad | **Poetry** (`pyproject.toml` + `poetry.lock`) | Nunca editar `poetry.lock` a mano; usar `poetry add` |
| Servicios (MLflow+Prefect) | **Docker Compose** en `services/` | `docker compose up -d`, requiere Docker Desktop abierto |
| Python | 3.12 – 3.14 | Instalado: 3.13.3 |

No hay despliegue en VM/nube para las actividades 1-3: todo corre local, los servicios se acceden por `localhost` (Docker mapea los puertos del contenedor directo al host, no hace falta averiguar IPs).

## Estructura del repo

```
Actividades-mlops/
├── AGENTS.md
├── README.md
├── .gitignore
├── services/                      # MLflow + Prefect (Docker Compose), compartido entre actividades
│   ├── docker-compose.yml
│   ├── requirements.txt           # alternativa sin Docker
│   └── README.md
└── actividad-1/
    ├── configs/config.yaml        # datos, modelo, MLflow/Prefect — nada hardcodeado en el código
    ├── data/                      # dataset (gitignored, .gitkeep solo)
    ├── flows/                     # flow de Prefect: cargar → entrenar → resumen
    ├── src/                       # código del caso de uso
    ├── notebooks/
    ├── tests/
    ├── pyproject.toml
    ├── poetry.lock
    └── README.md
```

Las próximas actividades (`actividad-2/`, `actividad-3/`, `actividad-4/`) deben seguir el mismo patrón de carpetas.

## Convenciones a seguir

- **Config fuera del código:** cualquier valor de datos/modelo/umbral va en `configs/config.yaml` de cada actividad, no hardcodeado.
- **Reproducibilidad:** semillas fijas (`random_state`), versiones fijadas vía Poetry — el criterio de éxito del curso es "otra persona clona el repo, instala el entorno y reproduce el experimento".
- **No versionar datasets crudos** ni artefactos de MLflow/Prefect (`mlflow_store/`, `.prefect/`, `*.csv` en `data/`) — ya está en `.gitignore`.
- **Cada actividad es autocontenida**: su propio `pyproject.toml`/`poetry.lock`, porque el profesor clona y evalúa *este* repo, no `../actividades/`.
- Idioma de documentación y commits: español, consistente con el resto del repo.

## Flujo de Git — nunca hacer push directo a `master`

`master` tiene **branch protection** en GitHub: push directo bloqueado (incluso para el admin), force-push y borrado de rama deshabilitados. Todo cambio entra por Pull Request ("GitHub Flow"):

```powershell
git checkout master && git pull
git checkout -b feature/nombre-descriptivo
# ... cambios, commits ...
git push -u origin feature/nombre-descriptivo
gh pr create --title "..." --body "..."
gh pr merge --merge --delete-branch   # solo tras confirmar con el usuario
```

Un agente de IA que edite este repo debe seguir este flujo: crear rama `feature/...`, nunca commitear directo en `master`. Fusionar un PR es una acción visible/compartida — confirmar con el usuario antes de hacer `gh pr merge`.

## Obligatorio vs. opcional en cada actividad

El profesor entrega la guía de cada actividad en pasos, y **no todos son obligatorios para cerrarla** — hay que leer el entregable final, no asumir que "más es mejor". En actividad 1, por ejemplo, el paso 3 (`flows/prueba_flow.py`: valida que Prefect+MLflow responden, experimento `prueba-validacion`) es obligatorio; el paso 4 (`flows/experimento_flow.py`: pipeline completa del caso de uso) está marcado explícitamente como opcional/recomendado. Un agente que construya una actividad debe verificar contra el texto literal de la guía qué es entregable obligatorio antes de darla por cerrada — no basta con que el código funcione o se vea completo. Ver [[feedback-scope-style]].

## Colaboradores en GitHub

Repo público: `https://github.com/KingAegon19/Actividades-mlops`. El profesor (`henrycala`) no es colaborador — puede ver y comentar directamente por ser público. Compañeros de equipo: pendientes de invitar (aún no se tienen sus usuarios de GitHub).

## Pendientes conocidos

- [x] Descargar el dataset de Kaggle a `actividad-1/data/` (hecho vía `kaggle auth login` + `kaggle datasets download`).
- [x] `src/` (carga de datos, modelo) y `flows/experimento_flow.py` para el caso de uso de default de tarjeta de crédito.
- [x] `flows/prueba_flow.py` (obligatorio): valida Prefect+MLflow, experimento `prueba-validacion`.
- [x] Dependencias fijadas con `==` exacto en `pyproject.toml` (no solo rangos), como pide la guía del profesor.
- [ ] Invitar compañeros de equipo como colaboradores cuando se tengan sus usuarios.
- [ ] Memoria 1 (usar la plantilla en `../actividades/docs/plantilla-memoria.md`, sin copiar contenido de la referencia).
