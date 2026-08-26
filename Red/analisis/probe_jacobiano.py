"""P1 y P2: mide el adjunto y el espectro de J_s sobre modelos ya entrenados.

P1  el solve adjunto (I - J_s)^T w = g_s agota o no sus `implicit_adjoint_iters`.
P2  espectro completo de J_s = dT/ds en el punto fijo (d=36 para n_x=3,N=2: se
    materializa con d productos vector-jacobiana, sin aproximar nada).

Replica exactamente lo que hace _ImplicitDR.backward, sin tocar el codigo del modelo.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                  # .../Red
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np, torch
torch.set_default_dtype(torch.float64); torch.set_num_threads(4)

from barrido.ood_eval import load_checkpoint
from barrido.data import load_cell, stack_cell
from red.backprop_implicit import _adjoint_solve
from entrenamiento.training import get_loss_fn

MODELOS = {
    "implicita (paper)":    "E1__actuadores__nx3__paper__implicit__impl__a0p01__sz150__s42",
    "desenrollado K=30":    "E1__actuadores__nx3__paper__unrolling__dr30__a0p01__sz150__s42",
}
BASE = Path(os.environ.get("LMINET_MODELS",
                           "analisis/resultados/barrido/E1/models"))
N_SYS = 40


def jacobiana_Js(model, s_star, y_hat, L, c, M_inv):
    """Materializa J_s = dT/ds en s* para CADA sistema del lote: (B, d, d).
    vjp(e_i) devuelve la fila i de J_s, y los sistemas no interactuan, asi que
    d productos vector-jacobiana bastan para todo el lote."""
    B, d = s_star.shape
    _, vjp_s = torch.func.vjp(lambda s: model._dr_state_step(s, y_hat, L, c, M_inv), s_star)
    filas = []
    for i in range(d):
        e = torch.zeros(B, d, dtype=s_star.dtype)
        e[:, i] = 1.0
        filas.append(vjp_s(e)[0])                       # (B, d) = fila i de J_s
    return torch.stack(filas, dim=1)                    # (B, d, d)


def sondear(nombre, run_id):
    ck = BASE / run_id / "epoch_400.pt"
    model, cfg, ep = load_checkpoint(ck)
    items = load_cell(cfg["n_x"], 1, 2)[:N_SYS]
    A, B = stack_cell(items)
    Bsz = A.shape[0]

    y_hat = model(A, B, return_unconstrained=True)
    L, c, M_inv = model._dr_precompute(A, B)
    with torch.no_grad():
        y_k, x_k = model._dr_iterate(y_hat.detach(), L, c, M_inv,
                                     model.implicit_max_iters, tol=model.implicit_tol)
        s_star = torch.cat([y_k, x_k], dim=1)
    d = s_star.shape[1]

    # ---- rhs REAL del adjunto: dL/dy* propagado por la proyeccion final ----
    def g_fn(yh, s):
        return model._dr_final_proj(yh, s[:, :model.dim_y], s[:, model.dim_y:], L, c, M_inv)
    y_star = g_fn(y_hat, s_star).detach().requires_grad_(True)
    Q, Y = model._y_to_matrices(y_star)
    perdida = get_loss_fn(cfg["loss"], model)(Q, Y, A, B)
    grad_out, = torch.autograd.grad(perdida, y_star)
    _, vjp_g = torch.func.vjp(g_fn, y_hat, s_star)
    _, g_s = vjp_g(grad_out)

    # ---- P1: el solve adjunto, tal cual lo corre el backward ----
    _, vjp_s = torch.func.vjp(lambda s: model._dr_state_step(s, y_hat, L, c, M_inv), s_star)
    w, used, res = _adjoint_solve(lambda v: vjp_s(v)[0], g_s,
                                  model.implicit_adjoint_iters,
                                  model.implicit_adjoint_tol, model.implicit_ridge)

    # ---- P1 por sistema (para ver la dispersion) ----
    usos, ress = [], []
    for b in range(Bsz):
        sb = s_star[b:b + 1]; yb = y_hat[b:b + 1]
        Lb, cb, Mb = L[b:b + 1], c[b:b + 1], M_inv[b:b + 1]
        _, vs = torch.func.vjp(lambda s: model._dr_state_step(s, yb, Lb, cb, Mb), sb)
        _, u, r = _adjoint_solve(lambda v: vs(v)[0], g_s[b:b + 1],
                                 model.implicit_adjoint_iters,
                                 model.implicit_adjoint_tol, model.implicit_ridge)
        usos.append(u); ress.append(r)
    usos = np.array(usos); ress = np.array(ress)

    # ---- P2: espectro completo de J_s ----
    J = jacobiana_Js(model, s_star, y_hat, L, c, M_inv).detach().numpy()
    rhos, n_uno, gaps = [], [], []
    for b in range(Bsz):
        lam = np.linalg.eigvals(J[b])
        mod = np.abs(lam)
        rhos.append(mod.max())
        n_uno.append(int((mod > 1 - 1e-6).sum()))
        m2 = np.sort(mod)[::-1]
        gaps.append(1.0 - m2[0])
    rhos = np.array(rhos); n_uno = np.array(n_uno); gaps = np.array(gaps)

    print(f"\n{'='*74}\n{nombre}   ({run_id})\n{'='*74}")
    print(f"  sistemas={Bsz}  d={d}  tope de iteraciones del adjunto="
          f"{model.implicit_adjoint_iters}  tol={model.implicit_adjoint_tol:g}  "
          f"ridge={model.implicit_ridge:g}")
    print(f"\n  P1 adjunto (lote completo): iters usadas={used}  residual final={res:.3e}"
          f"   {'AGOTA EL TOPE' if used >= model.implicit_adjoint_iters else 'converge'}")
    print(f"  P1 por sistema: agotan el tope {int((usos >= model.implicit_adjoint_iters).sum())}"
          f"/{Bsz}   iters mediana={np.median(usos):.0f}")
    print(f"                  residual final: mediana={np.median(ress):.2e}  "
          f"max={ress.max():.2e}   (tol pedida {model.implicit_adjoint_tol:g})")
    print(f"\n  P2 espectro de J_s: rho(J_s) mediana={np.median(rhos):.9f}  "
          f"min={rhos.min():.9f}  max={rhos.max():.9f}")
    print(f"      brecha 1-rho:   mediana={np.median(gaps):.3e}  min={gaps.min():.3e}")
    print(f"      autovalores con |lam|>1-1e-6: mediana={np.median(n_uno):.0f} de {d}"
          f"   (min={n_uno.min()}, max={n_uno.max()})")
    print(f"\n  Ganancia efectiva del modo lento:")
    print(f"      exacta 1/(1-rho)        ~ {1/max(np.median(gaps),1e-16):.3g}")
    print(f"      implicita (= iters usadas) ~ {np.median(usos):.0f}")
    print(f"      desenrollado K=30          = 30")


if __name__ == "__main__":
    for nombre, rid in MODELOS.items():
        sondear(nombre, rid)
