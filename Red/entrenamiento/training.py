"""
training.py
===========
Utilidades del flujo de datos -> entrenamiento para el encoder Deep Sets
(N-invariante). Reúne la transformación de datos, la construcción de UNA
arquitectura por orden n_x, el entrenamiento y la evaluación.

Núcleo del que depende:
  - data_loader.RobustControlMatlabDataset  (lectura/normalización de la BD)
  - lmi_net_deepsets.LMINetDeepSets          (arquitectura Deep Sets + solver DR)
"""

import numpy as np
import torch
from torch.utils.data import random_split

from pipeline.data_loader import RobustControlMatlabDataset
from red.lmi_net_deepsets import LMINetDeepSets

MAT_FILE = "DB_ssf_RS_500_c.mat"


# --------------------------- Carga y transformación ------------------------
def load_vertices(order, inputs, vertices, mat=MAT_FILE, limit=None):
    """Devuelve una lista de sistemas como tuplas (A, B) en float64.
    A: (N, n_x, n_x),  B: (N, n_x, n_u).  La normalización física la hace
    el propio RobustControlMatlabDataset."""
    ds = RobustControlMatlabDataset(mat, order=order, inputs=inputs, vertices=vertices)
    n = len(ds) if limit is None else min(limit, len(ds))
    items = []
    for i in range(n):
        _, _, A, B = ds[i]
        items.append((A.double(), B.double()))
    return items


def split_items(items, frac=0.8, seed=42):
    """Partición train/test reproducible."""
    n_tr = int(frac * len(items))
    tr, te = random_split(items, [n_tr, len(items) - n_tr],
                          generator=torch.Generator().manual_seed(seed))
    return list(tr), list(te)


def make_batches(items, batch=16, shuffle=True, rng=None, device="cpu"):
    """Agrupa items del MISMO N en lotes apilados (A,B). Cada lote es uniforme en N."""
    idx = list(range(len(items)))
    if shuffle:
        (rng or np.random).shuffle(idx)
    out = []
    for s in range(0, len(idx), batch):
        ch = idx[s:s + batch]
        A = torch.stack([items[i][0] for i in ch]).to(device)
        B = torch.stack([items[i][1] for i in ch]).to(device)
        out.append((A, B))
    return out


# --------------------------- Pérdida y modelo ------------------------------
def control_loss(Q, Y):
    """Costo auto-supervisado: volumen (traza Q) + esfuerzo de control (||Y||_F^2)."""
    vol = torch.diagonal(Q, dim1=-2, dim2=-1).sum(-1).mean()
    eff = (torch.linalg.matrix_norm(Y, ord="fro", dim=(1, 2)) ** 2).mean()
    return vol + 0.1 * eff


def build_model(order, inputs=1, alpha=0.01, dr_iters=30, **kw):
    """Crea UNA arquitectura Deep Sets para un orden n_x dado (N-invariante)."""
    return LMINetDeepSets(n=order, m=inputs, alpha=alpha, dr_iters=dr_iters, **kw).double()


# --------------------------- Entrenamiento ---------------------------------
def train(model, train_items_by_N, epochs=20, lr=1e-3, batch=16,
          seed=42, device="cpu", log_every=5):
    """Entrena el modelo de un orden sobre TODOS sus N (bucket por N: cada lote
    es uniforme en vértices, pero las épocas mezclan todos los N)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        batches = []
        for N, items in train_items_by_N.items():
            batches += make_batches(items, batch, shuffle=True, rng=rng, device=device)
        rng.shuffle(batches)
        run, nb = 0.0, 0
        for A, B in batches:
            opt.zero_grad()
            Q, Y = model(A, B)
            loss = control_loss(Q, Y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item(); nb += 1
        hist.append(run / max(nb, 1))
        if log_every and (ep == 0 or (ep + 1) % log_every == 0):
            print(f"  época {ep + 1:>3}/{epochs}  loss={hist[-1]:.4f}")
    return hist


# --------------------------- Evaluación ------------------------------------
@torch.no_grad()
def evaluate(model, items, order, alpha=0.01, dr_eval=2000, batch=16, device="cpu"):
    """% de sistemas estabilizados (Re(λ)<0), % que cumple decay-rate y peor
    autovalor de lazo cerrado promedio, sobre un conjunto de un mismo N."""
    old = model.dr_iters
    model.dr_iters = dr_eval
    model.eval()
    worst, stable = [], 0
    for A, B in make_batches(items, batch, shuffle=False, device=device):
        Q, Y = model(A, B)
        for k in range(A.shape[0]):
            Kp = Y[k] @ torch.linalg.inv(Q[k])
            we = -float("inf")
            for i in range(A.shape[1]):
                Acl = A[k, i] + B[k, i] @ Kp
                we = max(we, torch.max(torch.linalg.eigvals(Acl).real).item())
            worst.append(we)
            stable += int(we < 0.0)
    model.dr_iters = old
    tot = len(worst)
    return dict(
        stable_pct=100.0 * stable / tot,
        decay_pct=100.0 * sum(1 for e in worst if e <= -alpha) / tot,
        worst_mean=float(np.mean(worst)),
        n=tot,
    )
