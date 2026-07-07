"""
test_implicit_grad_matches_jacrev.py  (Parte B)
===============================================
Verifica que el nuevo backward MATRIX-FREE de la diferenciacion implicita coincide,
hasta ~1e-6, con la REFERENCIA por jacrev (Jacobianas densas + solve directo). Ambos
comparten el MISMO punto fijo s* y el MISMO ridge, asi que la unica diferencia bajo
prueba es "VJP + punto fijo" vs "materializar J_s, J_y y resolver denso".

La referencia jacrev es la implementacion de contraste que pedimos conservar: si el
matrix-free se desvia de ella, hay un bug en el adjunto.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]        # .../Red
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
from torch.func import vmap, jacrev

torch.set_default_dtype(torch.float64)

from entrenamiento.training import load_vertices          # noqa: E402
from red.actuators import LMINetActuators                 # noqa: E402
from red.backprop_implicit import _ImplicitDR             # noqa: E402


def _load_batch(n=3, m=1, N=2, k=6):
    items = load_vertices(n, m, N, limit=k)
    A = torch.stack([it[0] for it in items])
    B = torch.stack([it[1] for it in items])
    return A, B


def _reference_grad(model, y_hat, s_star, L, c, M_inv, grad_out):
    """Gradiente de REFERENCIA por jacrev (Jacobianas densas + solve directo).
        dy*/dy_hat = dg/dy_hat + dg/ds* (I - J_s)^{-1} J_y ,
    con el MISMO ridge que el matrix-free. Devuelve grad_yhat (B, dim_y)."""
    dim_y = model.dim_y
    sigma = model.sigma

    def step_single(s, yh, L1, c1, Mi):
        return model._dr_step_single(s, yh, L1, c1, Mi)

    def g_single(yh, s, L1, c1, Mi):                       # proyeccion final (una muestra)
        yk = s[:dim_y]; xk = s[dim_y:]
        s2 = 2 * sigma
        y_avg = (s2 * yh + yk) / (s2 + 1.0)
        return Mi @ (y_avg - L1.t() @ (c1 - xk))

    J_s = vmap(jacrev(step_single, argnums=0))(s_star, y_hat, L, c, M_inv)   # (B,d,d)
    J_y = vmap(jacrev(step_single, argnums=1))(s_star, y_hat, L, c, M_inv)   # (B,d,dim_y)
    dg_dyh = vmap(jacrev(g_single, argnums=0))(y_hat, s_star, L, c, M_inv)   # (B,dim_y,dim_y)
    dg_ds = vmap(jacrev(g_single, argnums=1))(y_hat, s_star, L, c, M_inv)    # (B,dim_y,d)

    B_sz, d = s_star.shape
    I = torch.eye(d, dtype=s_star.dtype).expand(B_sz, d, d)
    go = grad_out.unsqueeze(-1)
    g_s = torch.bmm(dg_ds.transpose(1, 2), go).squeeze(-1)                   # (B,d)
    g_yh = torch.bmm(dg_dyh.transpose(1, 2), go).squeeze(-1)                 # (B,dim_y)
    A_adj = (I - J_s).transpose(1, 2) + model.implicit_ridge * I
    w = torch.linalg.solve(A_adj, g_s.unsqueeze(-1)).squeeze(-1)
    return g_yh + torch.bmm(J_y.transpose(1, 2), w).squeeze(-1)


def test_implicit_grad_matches_jacrev(tol=1e-6):
    torch.manual_seed(0)
    A, B = _load_batch()
    model = LMINetActuators(n=3, alpha=0.01).double().eval()
    model.implicit_ridge = 1e-6
    model.implicit_adjoint_iters = 4000
    model.implicit_adjoint_tol = 1e-12

    # forward comun: un unico punto fijo s* que usan AMBOS caminos
    y_hat = torch.randn(A.shape[0], model.dim_y) * 0.1
    L, c, M_inv = model._dr_precompute(A, B)
    with torch.no_grad():
        y_k, x_k = model._dr_iterate(y_hat, L, c, M_inv, 6000, tol=1e-13)
        s_star = torch.cat([y_k, x_k], dim=1)

    grad_out = torch.randn(A.shape[0], model.dim_y)       # cotangente sobre y*

    # NUEVO: matrix-free (via autograd sobre _ImplicitDR)
    yh = y_hat.clone().requires_grad_(True)
    y_star = _ImplicitDR.apply(yh, s_star, L, c, M_inv, model)
    (y_star * grad_out).sum().backward()
    grad_new = yh.grad.clone()

    # REFERENCIA: jacrev denso
    grad_ref = _reference_grad(model, y_hat, s_star, L, c, M_inv, grad_out)

    err = (grad_new - grad_ref).abs().max().item()
    rel = err / max(grad_ref.abs().max().item(), 1e-12)
    print(f"[test_implicit_grad] max|dg|={err:.2e}  rel={rel:.2e}  (tol={tol:.0e})")
    assert err < tol or rel < tol, f"gradiente matrix-free != jacrev: abs={err:.2e} rel={rel:.2e}"


if __name__ == "__main__":
    test_implicit_grad_matches_jacrev()
    print("OK  test_implicit_grad_matches_jacrev")
