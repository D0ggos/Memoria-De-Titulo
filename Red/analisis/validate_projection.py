"""
validate_projection.py
======================

Banco de validacion numerica de la capa de proyeccion LMI.

Para cada sistema politopico (A_poly, B_poly) muestreado del dataset:

  1. Genera un punto inicial y_hat aleatorio (semilla fija por sistema).
  2. Resuelve la proyeccion EXACTA  y* = argmin ||y - y_hat||^2  s.a.  F_i(y) >= 0
     usando CVXPY/SCS (verdad de terreno).
  3. Ejecuta el forward de Douglas-Rachford (tal cual en lmi_net.py)
     y de Dykstra (con las dos simplificaciones especificas del problema)
     para una grilla de iteraciones.
  4. Reporta:
       - Distancia al optimo   ||y_k - y*_CVXPY||
       - Factibilidad          min_i lambda_min(F_i(y_k))
       - Curva de convergencia vs iteraciones
       - Tiempo de pared

Salidas:
  - results_projection.csv     (tabla larga: metodo, iters, sistema, metricas)
  - convergence_curves.png     (subplots: distancia y factibilidad vs iters)

Uso:
  python validate_projection.py --n-systems 10 --mat DB_ssf_RS_500_c.mat
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raíz Red/ en sys.path para correr como script

import numpy as np
import torch
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd

from pipeline.data_loader import RobustControlMatlabDataset
from red.lmi_net import LMINet


# ---------------------------------------------------------------------------
# Utilidades de construccion del operador F
# ---------------------------------------------------------------------------
def build_operator(model: LMINet, A_b: torch.Tensor, B_b: torch.Tensor):
    """Devuelve L, c, M_inv para un batch (B, ...) usando el helper de LMINet."""
    L, c = model._construct_L_c(A_b, B_b)
    B_sz = A_b.shape[0]
    I_y = torch.eye(model.dim_y, dtype=A_b.dtype).unsqueeze(0).expand(B_sz, -1, -1)
    M_inv = torch.linalg.inv(I_y + torch.bmm(L.transpose(1, 2), L))
    return L, c, M_inv


def feasibility_min_eig(model: LMINet, y: torch.Tensor,
                        A_b: torch.Tensor, B_b: torch.Tensor) -> float:
    """min sobre todos los bloques F^(i)(y) del menor autovalor (>=0 == factible)."""
    F_list = model._compute_F(y, A_b, B_b)
    mins = []
    for F in F_list:
        F_sym = 0.5 * (F + F.transpose(1, 2))
        mins.append(torch.linalg.eigvalsh(F_sym).min().item())
    return min(mins)


# ---------------------------------------------------------------------------
# Forward de Douglas-Rachford (replica de lmi_net.forward, parametrizado por y_hat)
# ---------------------------------------------------------------------------
def forward_dr(model: LMINet, y_hat: torch.Tensor,
               L: torch.Tensor, c: torch.Tensor, M_inv: torch.Tensor,
               n_iters: int, sigma: float) -> torch.Tensor:
    B_sz = y_hat.shape[0]
    block = model.n * model.n
    n_blocks = model.N + 1

    y_k = y_hat.clone()
    x_k_vec = (torch.bmm(L, y_k.unsqueeze(-1)).squeeze(-1) + c).clone()

    for _ in range(n_iters):
        # --- Proyeccion afin C1 con sesgo proximal (sigma) ---
        y_avg = (2 * sigma * y_hat + y_k) / (2 * sigma + 1.0)
        residuo = c - x_k_vec
        term2 = torch.bmm(L.transpose(1, 2), residuo.unsqueeze(-1)).squeeze(-1)
        y_w = torch.bmm(M_inv, (y_avg - term2).unsqueeze(-1)).squeeze(-1)
        x_w_vec = torch.bmm(L, y_w.unsqueeze(-1)).squeeze(-1) + c

        # --- Reflexion y proyeccion C2 (cono PSD) ---
        y_v = 2 * y_w - y_k
        x_v_blocks = []
        idx = 0
        for _i in range(n_blocks):
            Xi = (2 * x_w_vec[:, idx:idx + block] - x_k_vec[:, idx:idx + block]).view(B_sz, model.n, model.n)
            Xi = 0.5 * (Xi + Xi.transpose(1, 2))
            lam, V = torch.linalg.eigh(Xi)
            lam = torch.relu(lam)
            Xi_proj = torch.bmm(V, torch.bmm(torch.diag_embed(lam), V.transpose(1, 2)))
            x_v_blocks.append(Xi_proj.reshape(B_sz, -1))
            idx += block
        x_v_vec = torch.cat(x_v_blocks, dim=1)

        # --- Promediado ---
        y_k = y_v - y_w + y_k
        x_k_vec = x_v_vec - x_w_vec + x_k_vec

    # Proyeccion final sobre C1
    y_avg = (2 * sigma * y_hat + y_k) / (2 * sigma + 1.0)
    residuo = c - x_k_vec
    term2 = torch.bmm(L.transpose(1, 2), residuo.unsqueeze(-1)).squeeze(-1)
    y_star = torch.bmm(M_inv, (y_avg - term2).unsqueeze(-1)).squeeze(-1)
    return y_star


# ---------------------------------------------------------------------------
# Forward de Dykstra (con las dos simplificaciones del problema)
# ---------------------------------------------------------------------------
def forward_dykstra(model: LMINet, y_hat: torch.Tensor,
                    L: torch.Tensor, c: torch.Tensor, M_inv: torch.Tensor,
                    n_iters: int) -> torch.Tensor:
    B_sz = y_hat.shape[0]
    block = model.n * model.n
    n_blocks = model.N + 1

    y_curr = y_hat.clone()
    x_curr = (torch.bmm(L, y_curr.unsqueeze(-1)).squeeze(-1) + c).clone()
    q = torch.zeros_like(x_curr)  # unica correccion (cono); la afin se anula

    for _ in range(n_iters):
        # Proyeccion afin C1 (sin correccion)
        rhs = y_curr + torch.bmm(L.transpose(1, 2), (x_curr - c).unsqueeze(-1)).squeeze(-1)
        y_aff = torch.bmm(M_inv, rhs.unsqueeze(-1)).squeeze(-1)
        x_aff = torch.bmm(L, y_aff.unsqueeze(-1)).squeeze(-1) + c

        # Proyeccion C2 (cono PSD) con correccion
        x_in = x_aff + q
        proj_blocks = []
        idx = 0
        for _i in range(n_blocks):
            Xi = x_in[:, idx:idx + block].view(B_sz, model.n, model.n)
            Xi = 0.5 * (Xi + Xi.transpose(1, 2))
            lam, V = torch.linalg.eigh(Xi)
            lam = torch.relu(lam)
            Xi_proj = torch.bmm(V, torch.bmm(torch.diag_embed(lam), V.transpose(1, 2)))
            proj_blocks.append(Xi_proj.reshape(B_sz, -1))
            idx += block
        x_cone = torch.cat(proj_blocks, dim=1)

        q = x_in - x_cone
        y_curr, x_curr = y_aff, x_cone

    return y_curr


# ---------------------------------------------------------------------------
# Verdad de terreno: proyeccion exacta via CVXPY/SCS
# ---------------------------------------------------------------------------
def cvxpy_projection(model: LMINet, y_hat_np: np.ndarray,
                     A_np: np.ndarray, B_np: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Resuelve  min ||y - y_hat||^2  s.a.  F_i(y) >= 0  para un solo sistema.
    Devuelve (y_star, solve_time_s).
    """
    n, m, N = model.n, model.m, model.N
    alpha, eps = model.alpha, model.epsilon

    Q = cp.Variable((n, n), symmetric=True)
    Y = cp.Variable((m, n))

    # vec(y) en el MISMO orden que _y_to_matrices:
    #   primero los n*(n+1)/2 coefs del triangulo superior de Q (filas, recorrido upper-tri)
    #   luego los m*n coefs de Y aplanados (row-major)
    triu = np.triu_indices(n)
    y_Q_expr = cp.hstack([Q[i, j] for i, j in zip(triu[0], triu[1])])
    y_Y_expr = cp.reshape(Y, (m * n,), order='C')
    y_expr = cp.hstack([y_Q_expr, y_Y_expr])

    constraints = []
    for i in range(N):
        Ai, Bi = A_np[i], B_np[i]
        Fi = -(Ai @ Q + Q @ Ai.T + Bi @ Y + Y.T @ Bi.T + 2 * alpha * Q)
        constraints.append(Fi >> 0)
    constraints.append(Q - eps * np.eye(n) >> 0)

    objective = cp.Minimize(cp.sum_squares(y_expr - y_hat_np))
    prob = cp.Problem(objective, constraints)

    t0 = time.perf_counter()
    prob.solve(solver=cp.SCS, eps=1e-9, max_iters=200_000, verbose=False)
    dt = time.perf_counter() - t0

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"CVXPY no resolvio: status={prob.status}")

    y_star = np.concatenate([Q.value[triu], Y.value.flatten()])
    return y_star, dt


# ---------------------------------------------------------------------------
# Banco de pruebas principal
# ---------------------------------------------------------------------------
def run(args):
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(args.seed)

    dataset = RobustControlMatlabDataset(args.mat, order=args.n, inputs=args.m,
                                         vertices=args.N)
    n_total = len(dataset)
    if n_total == 0:
        raise SystemExit("Dataset vacio para esta topologia.")
    sample_idx = rng.choice(n_total, size=min(args.n_systems, n_total), replace=False)

    model = LMINet(n=args.n, m=args.m, N=args.N,
                   alpha=args.alpha, epsilon=args.epsilon,
                   dr_iters=1, sigma=args.sigma).double()

    iter_grid = [int(k) for k in args.iters.split(",")]
    rows = []

    for s_pos, idx in enumerate(sample_idx):
        features, _, A_t, B_t = dataset[int(idx)]
        A_b = A_t.unsqueeze(0).double()
        B_b = B_t.unsqueeze(0).double()

        # y_hat aleatorio reproducible (escala moderada)
        y_hat_np = rng.standard_normal(model.dim_y) * 0.3
        y_hat = torch.from_numpy(y_hat_np).unsqueeze(0)

        # Verdad de terreno
        try:
            y_star_np, t_cvx = cvxpy_projection(model, y_hat_np,
                                                A_b[0].numpy(), B_b[0].numpy())
        except RuntimeError as e:
            print(f"[sistema {idx}] saltado: {e}")
            continue
        y_star = torch.from_numpy(y_star_np).unsqueeze(0)
        feas_cvx = feasibility_min_eig(model, y_star, A_b, B_b)
        print(f"[{s_pos + 1}/{len(sample_idx)}] sistema {idx}: "
              f"CVXPY  ||y*||={np.linalg.norm(y_star_np):.3e}  "
              f"min_eig(F(y*))={feas_cvx:+.2e}  t={t_cvx:.2f}s")

        # Operador
        L, c, M_inv = build_operator(model, A_b, B_b)

        for k in iter_grid:
            # --- Douglas-Rachford ---
            t0 = time.perf_counter()
            y_dr = forward_dr(model, y_hat, L, c, M_inv, n_iters=k, sigma=args.sigma)
            t_dr = time.perf_counter() - t0
            dist_dr = float(torch.linalg.vector_norm(y_dr - y_star))
            feas_dr = feasibility_min_eig(model, y_dr, A_b, B_b)

            # --- Dykstra ---
            t0 = time.perf_counter()
            y_dk = forward_dykstra(model, y_hat, L, c, M_inv, n_iters=k)
            t_dk = time.perf_counter() - t0
            dist_dk = float(torch.linalg.vector_norm(y_dk - y_star))
            feas_dk = feasibility_min_eig(model, y_dk, A_b, B_b)

            rows.append(dict(system=int(idx), iters=k, method="DR",
                             dist=dist_dr, min_eig=feas_dr, time_s=t_dr))
            rows.append(dict(system=int(idx), iters=k, method="Dykstra",
                             dist=dist_dk, min_eig=feas_dk, time_s=t_dk))

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nResultados guardados en {args.out_csv}")
    print(df.groupby(["method", "iters"]).agg(
        dist_mean=("dist", "mean"), dist_med=("dist", "median"),
        feas_mean=("min_eig", "mean"), feas_min=("min_eig", "min"),
        time_mean=("time_s", "mean")).round(4))

    # Curvas de convergencia
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, sub in df.groupby("method"):
        agg = sub.groupby("iters").agg(d_med=("dist", "median"),
                                       f_med=("min_eig", "median")).reset_index()
        axes[0].plot(agg["iters"], agg["d_med"], marker="o", label=method)
        axes[1].plot(agg["iters"], agg["f_med"], marker="o", label=method)
    axes[0].set(xscale="log", yscale="log", xlabel="iteraciones",
                ylabel=r"mediana $\|y_k - y^*_{\mathrm{CVXPY}}\|$",
                title="Distancia al optimo")
    axes[1].axhline(0, color="k", lw=0.6, ls="--")
    axes[1].set(xscale="log", xlabel="iteraciones",
                ylabel=r"mediana $\min_i \lambda_{\min}(F_i(y_k))$",
                title="Factibilidad (>=0 == factible)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=140)
    print(f"Curvas guardadas en {args.out_png}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", default="DB_ssf_RS_500_c.mat")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--epsilon", type=float, default=1e-5)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--n-systems", type=int, default=10)
    p.add_argument("--iters", default="10,30,100,300,1000,3000",
                   help="lista de iteraciones separadas por coma")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", default="results_projection.csv")
    p.add_argument("--out-png", default="convergence_curves.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
