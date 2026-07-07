"""
bench_implicit_backward.py  (Verificación final #4)
===================================================
Mide la reduccion de tiempo del backward implicito MATRIX-FREE (VJP + punto fijo) frente al
jacrev que materializaba las Jacobianas completas J_s ∈ R^{dxd}, J_y, en un caso
representativo pesado (n_x=5, N=5, d = dim_y + (N+1)·n_x²). Ambos parten del MISMO punto
fijo s* y usan el MISMO ridge, asi que miden lo mismo; solo cambia como se resuelve el
adjunto. Tambien reporta el error del gradiente (deben coincidir ~1e-6).
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from torch.func import vmap, jacrev

torch.set_default_dtype(torch.float64)

from entrenamiento.training import load_vertices          # noqa: E402
from red.actuators import LMINetActuators                 # noqa: E402
from red.backprop_implicit import _ImplicitDR             # noqa: E402


def _reference_grad_timed(model, y_hat, s_star, L, c, M_inv, grad_out):
    """Gradiente por jacrev (materializa J_s, J_y) + solve directo. Devuelve (grad, seg)."""
    dim_y = model.dim_y; sigma = model.sigma
    t0 = time.perf_counter()

    def step_single(s, yh, L1, c1, Mi):
        return model._dr_step_single(s, yh, L1, c1, Mi)

    def g_single(yh, s, L1, c1, Mi):
        yk = s[:dim_y]; xk = s[dim_y:]
        s2 = 2 * sigma
        return Mi @ ((s2 * yh + yk) / (s2 + 1.0) - L1.t() @ (c1 - xk))

    J_s = vmap(jacrev(step_single, 0))(s_star, y_hat, L, c, M_inv)
    J_y = vmap(jacrev(step_single, 1))(s_star, y_hat, L, c, M_inv)
    dg_dyh = vmap(jacrev(g_single, 0))(y_hat, s_star, L, c, M_inv)
    dg_ds = vmap(jacrev(g_single, 1))(y_hat, s_star, L, c, M_inv)
    B_sz, d = s_star.shape
    I = torch.eye(d).expand(B_sz, d, d)
    go = grad_out.unsqueeze(-1)
    g_s = torch.bmm(dg_ds.transpose(1, 2), go).squeeze(-1)
    g_yh = torch.bmm(dg_dyh.transpose(1, 2), go).squeeze(-1)
    w = torch.linalg.solve((I - J_s).transpose(1, 2) + model.implicit_ridge * I, g_s.unsqueeze(-1)).squeeze(-1)
    grad = g_yh + torch.bmm(J_y.transpose(1, 2), w).squeeze(-1)
    return grad, time.perf_counter() - t0


def main(n=5, N=5, k=8, repeats=3):
    items = load_vertices(n, 1, N, limit=k)
    A = torch.stack([it[0] for it in items]); B = torch.stack([it[1] for it in items])
    model = LMINetActuators(n=n, alpha=0.01).double().eval()
    model.implicit_ridge = 1e-6
    model.implicit_adjoint_iters = 4000
    model.implicit_adjoint_tol = 1e-12

    y_hat = torch.randn(A.shape[0], model.dim_y) * 0.1
    L, c, M_inv = model._dr_precompute(A, B)
    with torch.no_grad():
        y_k, x_k = model._dr_iterate(y_hat, L, c, M_inv, 6000, tol=1e-13)
        s_star = torch.cat([y_k, x_k], dim=1)
    grad_out = torch.randn(A.shape[0], model.dim_y)
    d = s_star.shape[1]
    print(f"n_x={n} N={N} batch={A.shape[0]}  d={d} (dim_y={model.dim_y}, bloques={(N+1)*n*n})")

    # matrix-free (via autograd sobre _ImplicitDR)
    def mf():
        yh = y_hat.clone().requires_grad_(True)
        (_ImplicitDR.apply(yh, s_star, L, c, M_inv, model) * grad_out).sum().backward()
        return yh.grad

    t_mf = []
    for _ in range(repeats):
        t0 = time.perf_counter(); g_mf = mf(); t_mf.append(time.perf_counter() - t0)
    t_jr = []
    for _ in range(repeats):
        g_jr, dt = _reference_grad_timed(model, y_hat, s_star, L, c, M_inv, grad_out)
        t_jr.append(dt)

    err = (g_mf - g_jr).abs().max().item()
    tm, tj = min(t_mf), min(t_jr)
    print(f"matrix-free : {tm*1e3:8.1f} ms  (min de {repeats})")
    print(f"jacrev      : {tj*1e3:8.1f} ms  (min de {repeats})")
    print(f"speedup     : {tj/tm:6.2f}x")
    print(f"max|Δgrad|  : {err:.2e}  (debe ser ~1e-6)")


if __name__ == "__main__":
    main()
