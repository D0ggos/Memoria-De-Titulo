# Memoria de Título — Extensión politópica de LMI-Net

Síntesis de controladores robustos para sistemas politópicos mediante una red que propone
un certificado LMI y una capa de proyección Douglas–Rachford que lo vuelve factible. El
repositorio contiene el código, la base de datos de entrada y los resultados agregados del
estudio experimental que respalda el capítulo de resultados de la memoria.

## Estructura

```
Base de Datos/     base cruda de sistemas politópicos (.mat de MATLAB) y sus lectores
Red/               todo el código
  red/             arquitecturas (vanilla, vertices, actuadores) y solver DR + backprop
  entrenamiento/   carga de datos, las 6 funciones de pérdida y el bucle de entrenamiento
  pipeline/        ingesta y normalización de la base .mat
  barrido/         el estudio experimental: etapas, driver, evaluación, OOD, reportes
  analisis/        solucionador de referencia (CVXPY), benchmarks y reporte de la base
  tests/           pruebas del solver y del gradiente implícito
Escritura/         LaTeX de la memoria
legacy/            experimentos piloto que el barrido subsumió (ver legacy/README.md)
```

## Instalación

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r Red/requirements.txt
```

El estudio corre **en CPU**: la rueda solo-CPU de PyTorch es suficiente y bastante más
liviana (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). Las
dimensiones del problema son demasiado pequeñas para que la GPU compense la latencia de
lanzamiento de núcleos.

## Reproducir el estudio

Todos los comandos se lanzan desde `Red/`.

**1. Barrido completo** (E1–E4, 444 corridas). Es retomable: cada corrida escribe su shard
y el driver salta las ya hechas.

```bash
python -m barrido.driver --stage ALL --workers 12
```

Progreso en vivo, en otra terminal:

```bash
python -m barrido.estado --watch 60
```

**2. Evaluación fuera de distribución** (E5). No reentrena: evalúa los checkpoints ya
guardados sobre los dos bancos de deriva paramétrica.

```bash
python -m barrido.ood_eval --models analisis/resultados/barrido/E1/models
```

**3. Reporte y figuras.**

```bash
python -m barrido.reporte
python -m barrido.figuras --barrido analisis/resultados/barrido/E1 --ood analisis/resultados/barrido/OOD
```

**4. Comparación con el solucionador convexo** (sección Douglas–Rachford vs CVXPY):

```bash
python -m analisis.benchmark_dr_vs_cvxpy
```

**5. Pruebas:**

```bash
python tests/test_solver_equivalence.py
python tests/test_implicit_grad_matches_jacrev.py
python tests/test_losses_smoke.py
```

### Etapas del barrido

| Etapa | Ejes que varía | Corridas |
|---|---|---|
| E0 | prueba de humo del pipeline completo (pocas épocas) | 4 |
| E1 | pérdida × α × presupuesto de desenrollado, más la rama implícita | 240 |
| E2 | pérdida × orden n_x ∈ {2,3,4,5} | 48 |
| E3 | pérdida × tamaño de entrenamiento ∈ {50,150,400} | 36 |
| E4 | retropropagación (desenrollado vs implícita) y arquitectura × orden | 120 |
| E5 | evaluación OOD de los modelos ya entrenados (no entrena) | — |

## Costo

El barrido completo son **1 180 h-CPU** (1 022 h de entrenamiento + 158 h de evaluación),
que en el equipo de desarrollo —Ryzen 7 5700G de 8 núcleos, 32 GB, doce procesos
concurrentes— tomaron unos 4.6 días de reloj. La rama de diferenciación implícita
concentra el 66 % de ese cómputo pese a ser el 13 % de las corridas.

## Qué está versionado y qué no

Se versionan los resultados **agregados**: tiempos y memoria por corrida (`meta/`), curvas
de entrenamiento (`loss/`), la evaluación OOD completa, las figuras y los reportes. **No**
se versionan las filas de evaluación por sistema (2.1 GB) ni los checkpoints (832 MB):
ambos se regeneran con el paso 1. El resumen de todo el estudio está en
[`Red/analisis/resultados/barrido/REPORTE_BARRIDO.md`](Red/analisis/resultados/barrido/REPORTE_BARRIDO.md).
