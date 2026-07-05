"""
training.py
===========
Utilidades del flujo de datos -> entrenamiento para el encoder Deep Sets
(N-invariante). Reúne la transformación de datos, la construcción de UNA
arquitectura por orden n_x, el entrenamiento y la evaluación.

Núcleo del que depende:
  - data_loader.RobustControlMatlabDataset  (lectura/normalización de la BD)
  - vertices.LMINetVertices          (arquitectura Deep Sets + solver DR)
"""

import numpy as np
import torch
from torch.utils.data import random_split

from pipeline.data_loader import RobustControlMatlabDataset
from red.core import lmi_blocks

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


# --------------------------- Pérdidas --------------------------------------
def paper_loss(Q, Y, A_poly, B_poly, alpha=0.01, epsilon=1e-3, beta_soft=100.0):
    """Pérdida del paper LMI-Net (Ec. 21):
        L = c(y*) − β_soft · λ_min(F(y*)),   con  c(y*) = +logdet(Q).

    - `c = +logdet(Q)` (SIGNO POSITIVO): minimizar +logdet(Q) minimiza el volumen del
      elipsoide E(Q)={x : xᵀQ⁻¹x ≤ 1}. (El signo negativo era un error de una versión
      anterior; no se reintroduce.)
    - logdet defensivo: eigvalsh(Q).clamp_min(1e-12).log().sum, para no dar NaN con Q
      casi singular en épocas tempranas.
    - λ_min(F): F es bloque-diagonal (N+1 bloques); λ_min = mínimo sobre bloques.
      λ_min<0 penaliza violación residual; λ_min>0 premia margen interior. Ambos
      efectos son intencionales.
    - Se evalúa sobre la salida PROYECTADA (Q, Y), no sobre la predicción cruda.
    """
    logdetQ = torch.linalg.eigvalsh(Q).clamp_min(1e-12).log().sum(-1)              # (B,)
    F_list = lmi_blocks(Q, Y, A_poly, B_poly, alpha, epsilon)                      # N+1 bloques
    lam_min = torch.stack([torch.linalg.eigvalsh(0.5 * (F + F.transpose(-1, -2)))[..., 0]
                           for F in F_list], dim=-1).min(dim=-1).values            # (B,)
    return (logdetQ - beta_soft * lam_min).mean()


def control_loss(Q, Y, A_poly=None, B_poly=None):
    """Alternativa auto-supervisada (NO del paper): volumen (traza Q) + esfuerzo
    (||Y||_F^2). Misma firma que paper_loss; ignora A_poly, B_poly."""
    vol = torch.diagonal(Q, dim1=-2, dim2=-1).sum(-1).mean()
    eff = (torch.linalg.matrix_norm(Y, ord="fro", dim=(1, 2)) ** 2).mean()
    return vol + 0.1 * eff


def control_margen_loss(Q, Y, A_poly=None, B_poly=None, eta=1.0, piso=0.05):
    """control_loss + regularización del ŷ del encoder vía un PISO en λ_min(Q).

    PARA QUÉ SIRVE: 'control' y 'paper' premian Q pequeña (traza / logdet), así
    que el entrenamiento empuja al encoder hacia la frontera del conjunto
    factible, donde Q es casi singular. En distribución funciona; con sistemas
    fuera de distribución (politopos a mano, barridos de pole-shift) la
    proyección de ŷ aterriza en esa esquina mal condicionada: K = Y·Q⁻¹ explota,
    y el DR necesita miles de iteraciones o se estanca — aun cuando el DR puro
    desde ŷ ALEATORIO estabiliza esos mismos sistemas en ≤500 iteraciones (el
    solver no es el problema; la dirección de ŷ, sí).

    CÓMO: el término eta·relu(piso − λ_min(Q)) castiga certificados con Q casi
    singular y no toca a los que ya tienen margen (λ_min ≥ piso cuesta 0),
    manteniendo la ganancia K acotada.

    POR QUÉ LA PROBAMOS: hipótesis — entrenar con esto baja el iters_min y
    elimina el estancamiento en los barridos OOD (celda 4c del notebook) sin
    sacrificar % estabilizado en distribución. Comparar con grid_search
    {"loss": ["control", "control_margen"]}.
    """
    lam_min = torch.linalg.eigvalsh(Q)[..., 0]                     # (B,)
    margen = torch.relu(piso - lam_min).mean()
    return control_loss(Q, Y) + eta * margen


# --------------------- Registro de pérdidas por nombre ----------------------
# Permite elegir la pérdida por STRING (p.ej. desde CONFIG del notebook o un
# grid search) sin tocar el código de entrenamiento. Cada entrada tiene:
#   - "factory":   recibe el modelo (para leer alpha/epsilon si hace falta) y
#                  devuelve el loss_fn(Q, Y, A_poly, B_poly) que espera train().
#   - "direction": "min" si esa pérdida se quiere MINIMIZAR (por defecto) o "max"
#                  si se quiere MAXIMIZAR (p.ej. una tasa de decaimiento/margen).
#                  Importa para comparar contra CVXPY: CVXPY solo garantiza
#                  factibilidad, no optimiza ningún criterio, así que hace falta
#                  saber hacia dónde es "mejor" para decidir quién gana.
#
# Para añadir una pérdida nueva sin editar este archivo:
#   from entrenamiento.training import register_loss
#   register_loss("mi_loss", lambda model: mi_funcion_de_perdida, direction="min")
LOSS_REGISTRY = {
    "control": {"factory": lambda model: control_loss, "direction": "min"},
    "control_margen": {"factory": lambda model: control_margen_loss, "direction": "min"},
    "paper": {"factory": lambda model: (lambda Q, Y, A, B:
                        paper_loss(Q, Y, A, B, alpha=model.alpha, epsilon=model.epsilon)),
              "direction": "min"},
}


def register_loss(name, factory, direction="min"):
    """Registra una pérdida nueva bajo `name`. `factory(model) -> loss_fn(Q,Y,A,B)`.
    `direction`: "min" (minimizar, por defecto) o "max" (maximizar)."""
    if direction not in ("min", "max"):
        raise ValueError("direction debe ser 'min' o 'max'")
    LOSS_REGISTRY[name] = {"factory": factory, "direction": direction}


def get_loss_fn(name, model):
    """Resuelve una pérdida registrada para `model`. `name` puede ser un string
    del registro o ya un callable(Q,Y,A,B) (se devuelve tal cual)."""
    if callable(name):
        return name
    if name not in LOSS_REGISTRY:
        raise ValueError(f"pérdida '{name}' no registrada. Disponibles: "
                         f"{list(LOSS_REGISTRY)}. Registra la tuya con "
                         f"training.register_loss(nombre, factory, direction).")
    return LOSS_REGISTRY[name]["factory"](model)


def get_loss_direction(name, default="min"):
    """'min' o 'max': si la pérdida `name` se quiere minimizar o maximizar. Para
    un callable directo (no registrado), devuelve `default` (no hay forma de
    inferirla; pásala tú si tu loss_fn maximiza)."""
    if isinstance(name, str) and name in LOSS_REGISTRY:
        return LOSS_REGISTRY[name]["direction"]
    return default


# --------------------------- Entrenamiento ---------------------------------
def train(model, train_items_by_N, epochs=20, lr=1e-3, batch=16,
          seed=42, device="cpu", log_every=5, loss_fn=None):
    """Entrena el modelo sobre sus buckets (cada lote uniforme en (N, m)).
    `loss_fn(Q, Y, A_poly, B_poly)`: por DEFECTO la pérdida del paper (Ec. 21),
    enlazada a model.alpha/epsilon. Pasa `loss_fn=control_loss` para la alternativa
    auto-supervisada (volumen + esfuerzo)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model.to(device).train()
    if loss_fn is None:
        a, e = model.alpha, model.epsilon
        loss_fn = lambda Q, Y, A, B: paper_loss(Q, Y, A, B, alpha=a, epsilon=e)
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
            loss = loss_fn(Q, Y, A, B)
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
