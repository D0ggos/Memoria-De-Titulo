"""
benchmark_dr_vs_cvxpy.py
========================
Benchmark de velocidad de la capa de proyeccion Douglas-Rachford (DR, desenrollada)
frente al solver exacto CVXPY/SCS, barriendo N in {2,3,4,5} y presupuestos de
iteracion {500,1000,2000,3000,4000}.

Ademas identifica y GUARDA los sistemas que la DR no logra estabilizar
(certificado (Q,Y) tal que K=YQ^{-1} no deja Hurwitz al politopo) para analisis.

Reusa las rutinas ya validadas en validate_projection.py.

Salidas (en analisis/benchmark/):
  - speed.csv            velocidad y tasa de estabilizacion por (N, iters, metodo)
  - failures.csv         descriptores de los sistemas NO estabilizados por DR@max
  - failures.npz         matrices A,B crudas de esos sistemas (para reanalisis)
  - fig_speed.pdf        tiempo/sistema vs iteraciones (DR) con linea base CVXPY
  - fig_stab.pdf         tasa de estabilizacion vs iteraciones
  - fig_failures.pdf     descriptores: fallidos vs estabilizados
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # raiz Red/

import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt

from pipeline.data_loader import RobustControlMatlabDataset
from red.core import LMICore
from analisis.validate_projection import build_operator, cvxpy_projection

torch.set_default_dtype(torch.float64)

# ----------------------------- Config -------------------------------------
from entrenamiento.training import MAT_FILE as MAT      # ruta canonica de la base
OUT   = Path(__file__).resolve().parent / "resultados" / "proyeccion_dr_cvxpy"; OUT.mkdir(parents=True, exist_ok=True)
N_LIST   = [2, 3, 4, 5]
ITERS    = [500, 1000, 2000, 3000, 4000]
n, m     = 3, 1
ALPHA, EPS, SIGMA = 0.01, 1e-5, 0.01
N_DR   = 200          # sistemas por N para DR (estadistica de fallos)
N_CVX  = 50           # sistemas por N para timing CVXPY (mas lento)
SEED   = 0
TOL    = 0.0          # umbral de estabilizacion: abscisa de lazo cerrado < TOL

plt.rcParams.update({"font.family": "serif", "font.size": 10,
                     "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})


# --------------------- DR con checkpoints de iteracion --------------------
def dr_checkpoints(model, y_hat, L, c, M_inv, milestones, sigma):
    """Corre DR hasta max(milestones); en cada hito devuelve (y_proyectada, t_acumulado)."""
    B_sz = y_hat.shape[0]; block = model.n * model.n; n_blocks = model.N + 1
    Lt = L.transpose(1, 2)
    y_k = y_hat.clone()
    x_k = (torch.bmm(L, y_k.unsqueeze(-1)).squeeze(-1) + c).clone()

    def final_proj(y_k, x_k):
        y_avg = (2 * sigma * y_hat + y_k) / (2 * sigma + 1.0)
        term2 = torch.bmm(Lt, (c - x_k).unsqueeze(-1)).squeeze(-1)
        return torch.bmm(M_inv, (y_avg - term2).unsqueeze(-1)).squeeze(-1)

    out, ms = {}, set(milestones)
    t0 = time.perf_counter()
    for it in range(1, max(milestones) + 1):
        y_avg = (2 * sigma * y_hat + y_k) / (2 * sigma + 1.0)
        term2 = torch.bmm(Lt, (c - x_k).unsqueeze(-1)).squeeze(-1)
        y_w = torch.bmm(M_inv, (y_avg - term2).unsqueeze(-1)).squeeze(-1)
        x_w = torch.bmm(L, y_w.unsqueeze(-1)).squeeze(-1) + c
        y_v = 2 * y_w - y_k
        xv, idx = [], 0
        for _ in range(n_blocks):
            Xi = (2 * x_w[:, idx:idx + block] - x_k[:, idx:idx + block]).view(B_sz, model.n, model.n)
            Xi = 0.5 * (Xi + Xi.transpose(1, 2))
            lam, V = torch.linalg.eigh(Xi); lam = torch.relu(lam)
            xv.append(torch.bmm(V, torch.bmm(torch.diag_embed(lam), V.transpose(1, 2))).reshape(B_sz, -1))
            idx += block
        x_v = torch.cat(xv, dim=1)
        y_k = y_v - y_w + y_k
        x_k = x_v - x_w + x_k
        if it in ms:
            out[it] = (final_proj(y_k, x_k).clone(), time.perf_counter() - t0)
    return out


def per_system_metrics(model, y, A_b, B_b):
    """Por sistema: factibilidad LMI (min autovalor de los bloques) y abscisa de lazo cerrado."""
    Q, Y = model._y_to_matrices(y)
    F_list = model._compute_F(y, A_b, B_b)
    feas = torch.stack([torch.linalg.eigvalsh(0.5 * (F + F.transpose(1, 2))).min(dim=1).values
                        for F in F_list], 0).min(0).values.numpy()      # (B,)
    K = torch.bmm(Y, torch.linalg.inv(Q))                               # (B,m,n)
    cl = np.empty(A_b.shape[0])
    for b in range(A_b.shape[0]):
        cl[b] = max(torch.linalg.eigvals(A_b[b, i] + B_b[b, i] @ K[b]).real.max().item()
                    for i in range(model.N))
    return feas, cl


def descriptors(A_b, B_b, n, N):
    """Descriptores de sistema (crudos) para el analisis de fallos."""
    d = []
    for b in range(A_b.shape[0]):
        A = [A_b[b, i].numpy() for i in range(N)]; B = [B_b[b, i].numpy() for i in range(N)]
        Abar = np.mean(A, 0)
        d.append(dict(
            absc_ol=max(np.linalg.eigvals(Ai).real.max() for Ai in A),
            normA=max(np.linalg.norm(Ai, 2) for Ai in A),
            normB=max(np.linalg.norm(Bi, 2) for Bi in B),
            condA=max(np.linalg.cond(Ai) for Ai in A),
            dispA=max(np.linalg.norm(Ai - Abar, 2) for Ai in A),
        ))
    return pd.DataFrame(d)


# ------------------------------- Main -------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    speed_rows, fail_rows, fail_AB = [], [], []

    for N in N_LIST:
        ds = RobustControlMatlabDataset(MAT, order=n, inputs=m, vertices=N)
        idx = rng.choice(len(ds), size=min(N_DR, len(ds)), replace=False)
        A_b = torch.stack([ds[int(i)][2] for i in idx]).double()   # (B,N,n,n)
        B_b = torch.stack([ds[int(i)][3] for i in idx]).double()   # (B,N,n,m)
        B_sz = A_b.shape[0]

        model = LMICore(n=n, m=m, N=N, alpha=ALPHA, epsilon=EPS, dr_iters=1, sigma=SIGMA).double()
        Y_hat = torch.from_numpy(rng.standard_normal((B_sz, model.dim_y)) * 0.3)

        L, c, M_inv = build_operator(model, A_b, B_b)
        ckpts = dr_checkpoints(model, Y_hat, L, c, M_inv, ITERS, SIGMA)

        desc = descriptors(A_b, B_b, n, N)

        # --- DR: por hito de iteracion ---
        last_stab = None
        for it in ITERS:
            y_it, t_it = ckpts[it]
            feas, cl = per_system_metrics(model, y_it, A_b, B_b)
            stab = cl < TOL
            speed_rows.append(dict(N=N, iters=it, method="DR",
                                   time_per_sys=t_it / B_sz, time_total=t_it,
                                   stab_rate=float(stab.mean()),
                                   feas_rate=float((feas >= -1e-6).mean())))
            if it == ITERS[-1]:
                last_stab, last_cl, last_feas, last_y = stab, cl, feas, y_it

        # --- CVXPY: subconjunto, por sistema ---
        t_cvx = []
        for j in range(min(N_CVX, B_sz)):
            try:
                _, dt = cvxpy_projection(model, Y_hat[j].numpy(), A_b[j].numpy(), B_b[j].numpy())
                t_cvx.append(dt)
            except RuntimeError:
                pass
        speed_rows.append(dict(N=N, iters=np.nan, method="CVXPY",
                               time_per_sys=float(np.mean(t_cvx)), time_total=float(np.sum(t_cvx)),
                               stab_rate=1.0, feas_rate=1.0))

        # --- Fallos de DR@max: guardar ---
        desc["N"] = N; desc["ds_idx"] = idx
        desc["cl_absc_dr"] = last_cl; desc["feas_dr"] = last_feas; desc["stabilized"] = last_stab
        for b in np.where(~last_stab)[0]:
            fail_rows.append({**desc.iloc[b].to_dict()})
            fail_AB.append((N, int(idx[b]), A_b[b].numpy(), B_b[b].numpy()))
        # guardar tambien descriptores de TODOS (para comparar distribuciones)
        desc.to_csv(OUT / f"all_desc_N{N}.csv", index=False)
        print(f"N={N}: DR@{ITERS[-1]} estabiliza {last_stab.mean()*100:.1f}%  "
              f"({(~last_stab).sum()} fallos)  |  CVXPY {np.mean(t_cvx)*1e3:.1f} ms/sys  "
              f"DR {ckpts[ITERS[-1]][1]/B_sz*1e3:.2f} ms/sys")

    speed = pd.DataFrame(speed_rows); speed.to_csv(OUT / "speed.csv", index=False)
    fails = pd.DataFrame(fail_rows); fails.to_csv(OUT / "failures.csv", index=False)
    np.savez_compressed(OUT / "failures.npz",
                        meta=np.array([(N, i) for N, i, _, _ in fail_AB]),
                        **{f"A_{k}": AB[2] for k, AB in enumerate(fail_AB)},
                        **{f"B_{k}": AB[3] for k, AB in enumerate(fail_AB)})
    print(f"\nGuardado: {len(fails)} sistemas fallidos en {OUT}/failures.*")

    _plots(speed, fails, pd.concat([pd.read_csv(OUT / f"all_desc_N{N}.csv") for N in N_LIST]))


def _plots(speed, fails, alld):
    # fig 1: tiempo/sistema vs iteraciones (DR) + base CVXPY
    fig, ax = plt.subplots(figsize=(6.4, 4))
    dr = speed[speed.method == "DR"]
    for N in N_LIST:
        s = dr[dr.N == N]
        ax.plot(s.iters, s.time_per_sys * 1e3, marker="o", label=f"DR, N={N}")
    cvx = speed[speed.method == "CVXPY"]
    for N in N_LIST:
        ax.axhline(cvx[cvx.N == N].time_per_sys.iloc[0] * 1e3, ls="--", alpha=.5,
                   color="gray")
    ax.text(ITERS[0], cvx.time_per_sys.mean() * 1e3 * 1.1, "CVXPY (lineas punteadas)",
            fontsize=8, color="gray")
    ax.set(xlabel="iteraciones DR", ylabel="tiempo por sistema [ms]", yscale="log",
           title="Velocidad: DR (batcheado) vs CVXPY")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_speed.pdf"); plt.close(fig)

    # fig 2: tasa de estabilizacion vs iteraciones
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for N in N_LIST:
        s = dr[dr.N == N]
        ax.plot(s.iters, s.stab_rate * 100, marker="o", label=f"N={N}")
    ax.axhline(100, ls="--", color="k", alpha=.5, label="CVXPY (100%)")
    ax.set(xlabel="iteraciones DR", ylabel="% sistemas estabilizados",
           title="Estabilizacion vs presupuesto de iteraciones")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_stab.pdf"); plt.close(fig)

    # fig 3: descriptores fallidos vs estabilizados
    if len(fails):
        fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
        ok = alld[alld.stabilized == True]; bad = alld[alld.stabilized == False]
        for ax, col, t in [(axs[0], "condA", r"$\kappa(A)$"),
                           (axs[1], "dispA", "dispersión politopo"),
                           (axs[2], "absc_ol", "abscisa lazo abierto")]:
            ax.boxplot([np.log10(ok[col].clip(1e-3)) if col == "condA" else ok[col],
                        np.log10(bad[col].clip(1e-3)) if col == "condA" else bad[col]],
                       tick_labels=["estab.", "fallido"], showfliers=False, patch_artist=True)
            ax.set_title(("$\\log_{10}$ " if col == "condA" else "") + t)
        fig.suptitle("¿Qué caracteriza a los sistemas que la DR no estabiliza?", y=1.02)
        fig.tight_layout(); fig.savefig(OUT / "fig_failures.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
