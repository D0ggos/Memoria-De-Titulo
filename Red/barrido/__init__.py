"""
barrido/  —  barrido experimental por etapas de LMI-Net (Parte D)
================================================================
Cross-validation por etapas, con TODO guardado en crudo (una fila por punto de
resultado) para que el analisis y las figuras (Parte F) se hagan aparte sin recomputar.

Modulos:
  config.py      RunConfig (una corrida = una config completa) + generadores de etapas E0-E5.
  data.py        carga/particion de sistemas por (n_u, N); anillos A1 (visto) y A2 (retenido).
  metrics.py     metricas por-sistema del certificado (Q, Y) y techo CVXPY (opcional).
  evaluation.py  ladder de dr_eval por CHECKPOINT en UNA pasada; evalua un anillo.
  run.py         entrena UNA corrida con checkpoints de epoca; escribe filas crudas + modelos.
  driver.py      CLI + multiproceso a nivel de corrida sobre una etapa.

Principios (del enunciado):
  - Semilla es una COLUMNA; nunca se promedia en crudo.
  - Epocas y dr_eval NO multiplican el nº de entrenamientos: son checkpoints.
  - Etapas parametrizables; no hay una unica corrida gigante hardcodeada.
"""
