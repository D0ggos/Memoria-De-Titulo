# resultados/

Salidas de todos los benchmarks y experimentos. Tres orígenes, tres carpetas:

- **`benchmark/<arch>/<backprop>/n{n}_m{m}/`** — del benchmark general
  (`analisis/benchmark.py` → `run_experiment(..., save_failures=True)`, el que usa
  el notebook). **Un directorio por combinación** (arquitectura × backprop × orden ×
  actuadores). Dentro:
  - `failures.csv` — los sistemas NO estabilizados, con columnas
    `arch, backprop, n, m, N, worst_eig` + descriptores (κ(A), ‖B‖, dispersión…).
  - `failures.npz` — las matrices `A, B` crudas de esos sistemas.
  - `summary.csv` — % estabilizado por N de esa combinación.

- **`experimentos/`** — de los scripts `entrenamiento/experimento_*.py`
  (generalización cross-actuador, dr_budget, implícita vs unrolling, unroll_iters).

- **`proyeccion_dr_cvxpy/`** — del benchmark de velocidad de la capa de proyección
  `analisis/benchmark_dr_vs_cvxpy.py` (DR desenrollado vs CVXPY, init aleatorio).
