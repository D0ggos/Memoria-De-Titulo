"""
ood_banks.py  —  bancos fuera de distribucion (Parte E)
=======================================================
Dos familias parametricas de sistemas que la red NUNCA vio en entrenamiento, para medir
extrapolacion. Se construyen en fisico y se NORMALIZAN igual que la base (dividir por
gamma=max|A,B|): la normalizacion es un escalado uniforme (gamma>0) que NO cambia el
signo de Re λ, asi que "estable en normalizado <=> estable en fisico".

  - Pole-shift nominal (N=1, n_x=3):  A(s) = A0 + s·I,   s ∈ [−1, 2].
      A0 = [[0,1,0],[0,0,1],[2,−3,1]],  B = [0,0,1]ᵀ.  Como λ(A0+sI)=λ(A0)+s, el
      parametro s desplaza TODOS los polos en bloque hacia el semiplano derecho.

  - Politopo del profesor (N=2, n_x=2):  A_i(δ) = A_i⁰ + δ·D_i,   δ ∈ [0, 2].
      A1⁰=[[0,1],[−2,−2]], D1=diag(2,1);  A2⁰=[[0,1],[−2,−3]], D2=diag(−2,1);
      B=[0,1]ᵀ en ambos vertices. En δ=1, A1 tiene autovalores EXACTOS {0, +1}
      (el vertice 1 toca la frontera de estabilidad). δ mueve un vertice hacia la
      inestabilidad y el otro hacia adentro: la interseccion factible se estrecha.
"""
import numpy as np
import torch

from analisis.benchmark import polytope_from_vertices, shift_poles, normalize_system

# --------------------------- Pole-shift nominal ----------------------------
POLE_A0 = [[0, 1, 0], [0, 0, 1], [2, -3, 1]]
POLE_B = [[0], [0], [1]]
POLE_S = np.linspace(-1.0, 2.0, 31)                # s ∈ [−1, 2]

# --------------------------- Politopo del profesor -------------------------
PROF_A1 = [[0, 1], [-2, -2]]; PROF_D1 = [[2, 0], [0, 1]]
PROF_A2 = [[0, 1], [-2, -3]]; PROF_D2 = [[-2, 0], [0, 1]]
PROF_B = [[0], [1]]
PROF_DELTAS = np.linspace(0.0, 2.0, 21)           # δ ∈ [0, 2]


def pole_shift_system(s, normalize=True):
    """Devuelve (A_poly, B_poly) del pole-shift nominal a desplazamiento s. N=1, n_x=3."""
    A0, B0 = polytope_from_vertices([POLE_A0], [POLE_B])       # (1,3,3),(1,3,1)
    A = shift_poles(A0, float(s), directions=None)            # A0 + s·I
    if normalize:
        A, B0 = normalize_system(A, B0)
    return A, B0


def professor_system(delta, normalize=True):
    """Devuelve (A_poly, B_poly) del politopo del profesor a δ. N=2, n_x=2."""
    A0, B0 = polytope_from_vertices([PROF_A1, PROF_A2], [PROF_B])   # (2,2,2),(2,2,1)
    A = shift_poles(A0, float(delta), directions=[PROF_D1, PROF_D2])
    if normalize:
        A, B0 = normalize_system(A, B0)
    return A, B0


def pole_shift_bank(s_values=POLE_S, normalize=True):
    """Lista de (s, A_poly, B_poly)."""
    return [(float(s), *pole_shift_system(s, normalize)) for s in s_values]


def professor_bank(deltas=PROF_DELTAS, normalize=True):
    """Lista de (delta, A_poly, B_poly)."""
    return [(float(d), *professor_system(d, normalize)) for d in deltas]
