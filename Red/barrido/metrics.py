"""
metrics.py  —  metricas por-sistema del certificado (Q, Y) (Partes D/F)
=======================================================================
Todo POR SISTEMA (crudo): el agregado y las figuras se calculan despues. Nada se
promedia aqui. Reusa la construccion de F de core (lmi_blocks) para el residual.

Metricas:
  worst           max_i Re λ(A_i + B_i K)      (abscisa de lazo cerrado; <0 == estable)
  stable          worst < 0
  decay           worst <= -alpha              (cumple la tasa prescrita)
  margin          -worst                       (holgura; para violin/histograma)
  lam_min_F       λ_min(F(y*))                 (violacion residual de la LMI al presupuesto)
  lam_min_Q,kappaQ λ_min(Q), κ(Q)              (condicionamiento del certificado)
  normK2, normKf  ||K||_2, ||K||_F,  K = Y Q⁻¹
  proj_dist       ||ŷ − y*||                   (calidad del encoder, aislada del solver)
"""
import time

import numpy as np
import torch

from red.core import lmi_blocks


def _K_from_QY(Q, Y):
    """K = Y Q⁻¹ via solve (Q simetrica): Kᵀ = Q⁻¹ Yᵀ. Devuelve K (B,m,n)."""
    Kt = torch.linalg.solve(Q, Y.transpose(-1, -2))            # (B,n,m)
    return Kt.transpose(-1, -2)                                 # (B,m,n)


def closed_loop_worst(A_poly, B_poly, K):
    """Peor parte real del lazo cerrado por sistema (B,). Vectorizado sobre vertices:
    A_cl_i = A_i + B_i K,  eigvals batcheados."""
    Acl = A_poly + torch.matmul(B_poly, K.unsqueeze(1))        # (B,N,n,n)
    ev = torch.linalg.eigvals(Acl)                             # (B,N,n) complejo
    return ev.real.amax(dim=(1, 2))                            # (B,)


def lambda_min_F(model, y_star, A_poly, B_poly):
    """λ_min(F(y*)) = min sobre los N+1 bloques (violacion residual; >=0 == factible)."""
    Q, Y = model._y_to_matrices(y_star)
    F_list = lmi_blocks(Q, Y, A_poly, B_poly, model.alpha, model.epsilon)
    return torch.stack([torch.linalg.eigvalsh(0.5 * (F + F.transpose(-1, -2)))[..., 0]
                        for F in F_list], dim=-1).min(dim=-1).values


@torch.no_grad()
def certificate_metrics(model, y_star, A_poly, B_poly, y_hat=None):
    """Dict de arrays numpy (B,) con todas las metricas por-sistema del certificado y*."""
    Q, Y = model._y_to_matrices(y_star)
    K = _K_from_QY(Q, Y)
    worst = closed_loop_worst(A_poly, B_poly, K)
    evQ = torch.linalg.eigvalsh(Q)                             # (B,n) ascendente
    lamQ = evQ[..., 0]
    kappaQ = evQ[..., -1] / evQ[..., 0].clamp_min(1e-12)
    lamF = lambda_min_F(model, y_star, A_poly, B_poly)
    normK2 = torch.linalg.matrix_norm(K, ord=2, dim=(-2, -1))
    normKf = torch.linalg.matrix_norm(K, ord="fro", dim=(-2, -1))
    out = dict(
        worst=worst, stable=(worst < 0.0), decay=(worst <= -model.alpha),
        margin=-worst, lam_min_F=lamF, lam_min_Q=lamQ, kappa_Q=kappaQ,
        norm_K2=normK2, norm_Kf=normKf,
    )
    if y_hat is not None:
        out["proj_dist"] = torch.linalg.vector_norm(y_hat - y_star, dim=-1)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


# --------------------------- Techo CVXPY (opcional) ------------------------
def cvxpy_feasible_batch(model, A_poly, B_poly, y_hat, max_systems=30):
    """Techo/verdad-de-terreno: por sistema, ¿es factible la LMI? (¿existe certificado?).
    Proyecta ŷ con CVXPY/SCS: si resuelve, el conjunto es no vacio -> factible (y el K
    resultante estabiliza por construccion). Devuelve dict con arrays (feasible, t_ms) y el
    nº de sistemas evaluados. cvxpy se importa aqui para no exigirlo si no se usa."""
    from analisis.validate_projection import cvxpy_projection    # import diferido
    B_sz = min(max_systems, A_poly.shape[0])
    feas = np.zeros(B_sz, dtype=bool)
    tms = np.full(B_sz, np.nan)
    for j in range(B_sz):
        yj = y_hat[j].detach().cpu().numpy()
        try:
            t0 = time.perf_counter()
            _ = cvxpy_projection(model, yj, A_poly[j].cpu().numpy(), B_poly[j].cpu().numpy())
            tms[j] = (time.perf_counter() - t0) * 1e3
            feas[j] = True
        except Exception:
            feas[j] = False
    return dict(feasible=feas, cvxpy_t_ms=tms, n=B_sz)
