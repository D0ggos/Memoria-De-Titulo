"""
test_solver_equivalence.py  (Parte A)
=====================================
DEMUESTRA la unificacion de la salida de los dos solvers: a punto fijo (presupuesto
grande, tol chica) unrolling e implicita deben devolver el MISMO certificado (Q, Y),
porque ambos aterrizan con la proyeccion final Pi_{C1} antes de decodificar.

Con el mismo encoder (mismos pesos) el y_hat es identico; lo unico que cambia es el
solver. Si (Q,Y) coinciden hasta ~1e-6, la definicion de salida quedo unificada y
cualquier diferencia posterior unrolling-vs-implicita es atribuible SOLO al gradiente.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]        # .../Red
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                     # load_vertices busca el .mat aqui

import torch

torch.set_default_dtype(torch.float64)

from entrenamiento.training import load_vertices          # noqa: E402
from red.actuators import LMINetActuators                 # noqa: E402


def _load_batch(n=3, m=1, N=2, k=8):
    items = load_vertices(n, m, N, limit=k)
    A = torch.stack([it[0] for it in items])
    B = torch.stack([it[1] for it in items])
    return A, B


def test_solver_equivalence(tol=1e-6):
    torch.manual_seed(0)
    A, B = _load_batch()

    model = LMINetActuators(n=3, alpha=0.01, dr_iters=6000).double().eval()
    model.implicit_max_iters = 6000        # mismo presupuesto de convergencia
    model.implicit_tol = 1e-12

    with torch.no_grad():
        model.set_implicit(False)          # unrolling
        Qu, Yu = model(A, B)
        model.set_implicit(True)           # implicita
        Qi, Yi = model(A, B)

    dQ = (Qu - Qi).abs().max().item()
    dY = (Yu - Yi).abs().max().item()
    print(f"[test_solver_equivalence] max|dQ|={dQ:.2e}  max|dY|={dY:.2e}  (tol={tol:.0e})")
    assert dQ < tol, f"Q difiere entre solvers: {dQ:.2e}"
    assert dY < tol, f"Y difiere entre solvers: {dY:.2e}"


if __name__ == "__main__":
    test_solver_equivalence()
    print("OK  test_solver_equivalence")
