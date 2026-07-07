# Barrido experimental LMI-Net (Partes D–F)

Cross-validation por etapas con **todo guardado en crudo** (una fila por punto de
resultado). El análisis y las figuras se hacen **aparte**, leyendo los shards; nada se
recomputa. La semilla es una **columna**, nunca se promedia en crudo.

## Estructura

| módulo | rol |
|---|---|
| `config.py` | ejes, `RunConfig` (una corrida = una config), generadores de etapas `E0`–`E4` |
| `data.py` | celdas `(n_u, N)`, anillos **A1** (visto) y **A2** (retenido: `N=5`, `n_u=2`) |
| `metrics.py` | métricas por-sistema del certificado `(Q,Y)` + techo CVXPY (opcional) |
| `evaluation.py` | ladder de `dr_eval` por checkpoint en **una** pasada; evalúa un anillo |
| `run.py` | entrena UNA corrida con checkpoints de época; escribe filas crudas + modelos |
| `driver.py` | CLI + multiproceso a nivel de corrida sobre una etapa |
| `ood_banks.py` | bancos OOD: pole-shift nominal y politopo del profesor |
| `ood_eval.py` | **E5** (solo-eval de modelos entrenados) + instrumentación del mecanismo |
| `figuras.py` | figuras de entrenamiento (A1/A2) y del profesor (P1–P5) |

Épocas `{50,100,200,400}` y `dr_eval {100,…,8000}` se obtienen por **checkpoint**: NO
multiplican el nº de entrenamientos.

## Cómo correr

```bash
# 0) smoke test de punta a punta (rápido)
python -m barrido.driver --stage E0 --workers 4

# 1) factorial núcleo (el bloque grande); --cvxpy activa el techo CVXPY (lento)
python -m barrido.driver --stage E1 --workers 16
python -m barrido.driver --stage E2 --workers 16
python -m barrido.driver --stage E3 --workers 16
python -m barrido.driver --stage E4 --workers 16

# 5) OOD: SOLO evalúa los modelos ya entrenados (sin reentrenar)
python -m barrido.ood_eval --models analisis/resultados/barrido/E1/models \
                           --out    analisis/resultados/barrido/OOD

# figuras (lee los CSV/parquet; no recomputa)
python -m barrido.figuras --barrido analisis/resultados/barrido/E1 \
                          --ood     analisis/resultados/barrido/OOD \
                          --out     analisis/resultados/barrido/figuras
```

Salidas por etapa en `analisis/resultados/barrido/<E?>/`:
`results/` (una fila por sistema × anillo × época × dr_eval), `loss/` (loss por época),
`meta/` (tiempo/memoria por corrida), `models/` (checkpoints), `errors/` (trazas de fallo).

## Tests (Partes A/B/C)

```bash
python tests/test_solver_equivalence.py            # Parte A: unificación de salida
python tests/test_implicit_grad_matches_jacrev.py  # Parte B: VJP matrix-free == jacrev
python tests/test_losses_smoke.py                  # Parte C: 6 pérdidas sin NaN/Inf
python tests/bench_implicit_backward.py            # Verif. #4: speedup matrix-free vs jacrev
```

> **Nota de entorno.** Estos scripts necesitan el `.venv` con torch/numpy cargables. Si el
> intérprete no carga sus DLLs nativas por CLI (visto en Python 3.14.0a2 en Windows),
> córrelos desde el kernel de Jupyter del `.venv` o desde un entorno donde `import torch`
> funcione.
