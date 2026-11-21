# Evidencia: reportes de drift reales

Reportes HTML de Evidently generados en vivo (no simulados) por `drift_features_flow.py`, con
`inferencia.tamano_lote=500` -- ver [`../../PLAN.md`](../../PLAN.md) para el detalle completo
de la corrida y los números exactos.

- **`drift_features_nivel_0.0.html`** -- lote sin drift emulado (`nivel_drift: 0.0`). 0% de las
  variables marcadas con drift por Evidently; sin alerta (umbral 30%).
- **`drift_features_nivel_0.6.html`** -- mismo lote base, con drift emulado al 60%
  (`nivel_drift: 0.6`). 61% de las variables marcadas con drift; **alerta**.

Ambos reportes son archivos HTML autocontenidos -- se pueden abrir directamente en un navegador,
sin necesidad de levantar ningún servicio.
