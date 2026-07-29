# legacy/ — experimentos piloto

Scripts exploratorios que motivaron el diseño del barrido y que **el barrido E1–E4 ya
subsume**. Se conservan por trazabilidad; no hacen falta para reproducir el capítulo de
resultados y **sus imports están rotos** desde que se movieron aquí (esperan estar dentro
de `Red/analisis/` o `Red/entrenamiento/`). Para correr alguno, devuélvelo a su carpeta
original o lanza Python con `Red/` en el `PYTHONPATH`.

| Script | Qué hacía | Qué lo reemplazó |
|---|---|---|
| `experimento_reporte.py` | matriz completa de experimentos preliminar | etapas E1–E4 (`barrido/config.py`) |
| `experimento_hparams.py` | barrido ampliado de hiperparámetros | etapa E1 (factorial núcleo) |
| `experimento_unroll_iters.py` | barrido de iteraciones de desenrollado | eje `dr_train` de E1 |
| `experimento_implicito_vs_unroll.py` | duelo desenrollado vs implícita | eje `backprop` de E1/E4 |
| `experimento_actuadores_full.py` | invarianza de actuadores sobre toda la base | etapas E2 y E4 |
| `experimento_dr_budget.py` | recuperación de fallos con más iteraciones | sondeo `dr_eval` de `barrido/evaluation.py` |
| `experimento_trayectoria.py` | trayectoria del peor autovalor vs iteración | `professor_mechanism` en `barrido/ood_eval.py` |
| `experimento_plano_complejo.py` | mapa de polos en el plano complejo | figura del capítulo (no se regenera) |
| `experimento_vanilla_paper.py` | reproducción fiel de la receta del paper | arquitectura `vanilla` en E4 |
| `calibrar_tiempos.py` | estimación de costo antes de lanzar el barrido | `meta/` de cada corrida (tiempo y memoria reales) |
| `gpu_benchmark.py` | prueba de esfuerzo de la GPU | ninguno: el barrido corre en CPU |
| `pruebas.ipynb` | cuaderno de exploración | — |
