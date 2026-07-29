"""
data.py  —  carga de sistemas, celdas y anillos A1/A2 (Parte D)
===============================================================
Una CELDA es una topologia (n_u=m, N). El barrido entrena las arquitecturas INVARIANTES
sobre un conjunto de celdas y guarda otras RETENIDAS para probar generalizacion zero-shot:

  A1 (visto)      : test 20% de las celdas ENTRENADAS. Particion fija seed=42
                    (independiente de la semilla de la corrida).
  A2 (estructural): celdas RETENIDAS del entrenamiento — N=5 y n_u=2 nunca vistos.
                    Solo tienen sentido para arquitecturas invariantes.

Reglas por arquitectura:
  - actuadores  : invariante a N y a n_u. Entrena en (m=1, N in {2,3,4}); A2 = (m=1,N=5)
                  U (m=2, cualquier N).
  - vertices    : invariante a N, n_u FIJO (=1). Entrena en (m=1, N in {2,3,4});
                  A2 = (m=1, N=5). (No puede evaluar m=2: cambia la dimension de entrada.)
  - vanilla     : NO invariante (una red por (n_x, n_u, N)). Entrena en UNA celda
                  (m=1, N=2); A1 = 20% de esa celda; A2 = [] (no aplica).

`train_size` limita cuantos sistemas de la porcion de train (el 80%) se usan por celda.
"""
from functools import lru_cache
from pathlib import Path

import torch

from entrenamiento.training import MAT_FILE, load_vertices, split_items

ROOT = Path(__file__).resolve().parents[1]                 # .../Red
MAT = MAT_FILE                                              # unica fuente de verdad de la ruta

TRAIN_NS = (2, 3, 4)                                        # N vistos en entrenamiento
HELDOUT_N = 5                                               # N estructuralmente retenido
HELDOUT_M = 2                                               # n_u estructuralmente retenido
PROBE_MS = (1, 2)
PROBE_NS = (2, 3, 4, 5)


@lru_cache(maxsize=None)
def load_cell(n_x, m, N):
    """Items (A,B) de una celda (memoizado por proceso). Lista vacia si la celda no existe.
    Una celda ausente es normal (no toda combinacion (n_x,m,N) esta en la base); una base
    ausente NO lo es y se reporta, en vez de degradarse a 'no hay celdas de entrenamiento'."""
    if not Path(MAT).exists():
        raise FileNotFoundError(
            f"no encuentro la base de datos en {MAT}. Deberia estar en "
            f"'Base de Datos/DB_ssf_RS_500_c.mat' (ver README).")
    try:
        return load_vertices(n_x, m, N, mat=MAT)
    except Exception:
        return []


def available_cells(n_x):
    """{(m, N): n_sistemas} de las celdas NO vacias para este orden."""
    out = {}
    for m in PROBE_MS:
        for N in PROBE_NS:
            items = load_cell(n_x, m, N)
            if len(items):
                out[(m, N)] = len(items)
    return out


def _split_cell(items, train_size):
    """Particion fija seed=42: (train[:train_size], test20%). El test A1 NO depende de la
    semilla de la corrida (comparabilidad entre corridas)."""
    tr, te = split_items(items, frac=0.8, seed=42)
    return tr[:train_size], te


def make_datasets(arch, n_x, train_size):
    """Devuelve (train_by_cell, a1_by_cell, a2_by_cell), cada uno dict (m,N)->items.
    train_by_cell alimenta el entrenamiento (buckets uniformes en (m,N))."""
    cells = available_cells(n_x)
    train_by_cell, a1_by_cell, a2_by_cell = {}, {}, {}

    if arch == "vanilla":
        key = (1, 2)
        if key in cells:
            tr, te = _split_cell(load_cell(n_x, *key), train_size)
            train_by_cell[key] = tr
            a1_by_cell[key] = te
        return train_by_cell, a1_by_cell, a2_by_cell     # A2 no aplica a vanilla

    # arquitecturas invariantes (vertices / actuadores): entrenan en m=1, N in {2,3,4}
    for N in TRAIN_NS:
        key = (1, N)
        if key in cells:
            tr, te = _split_cell(load_cell(n_x, 1, N), train_size)
            train_by_cell[key] = tr
            a1_by_cell[key] = te

    # A2 estructural: N=5 (m=1) siempre; m=2 (cualquier N) SOLO para actuadores
    for (m, N), _ in cells.items():
        held = (N == HELDOUT_N) or (m == HELDOUT_M)
        if not held:
            continue
        if arch == "vertices" and m != 1:
            continue                                      # vertices no acepta m!=1
        a2_by_cell[(m, N)] = load_cell(n_x, m, N)

    return train_by_cell, a1_by_cell, a2_by_cell


def stack_cell(items, device="cpu", dtype=None):
    """Apila una celda (uniforme en (m,N)) a tensores (B,N,n,n),(B,N,n,m).
    `device`/`dtype`: destino (dtype=None conserva float64 de los items)."""
    A = torch.stack([it[0] for it in items]).to(device=device, dtype=dtype)
    B = torch.stack([it[1] for it in items]).to(device=device, dtype=dtype)
    return A, B
