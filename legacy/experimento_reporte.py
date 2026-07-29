"""
experimento_reporte.py
======================
Corre la MATRIZ COMPLETA de experimentos para la seccion de resultados de la
tesis y guarda tablas (CSV) + figuras (PNG para el reporte, PDF para LaTeX) en
analisis/resultados/reporte/.

Bloques (cada uno = una subseccion de resultados):
  A  Perdidas (control / control_margen / paper) en distribucion + vs CVXPY
  B  Backprop: unrolling vs diferenciacion implicita (mismo init/datos)
  C  Escalera de arquitecturas: vanilla (paper) / vertices / actuadores
  D  Iteraciones DR desenrolladas en entrenamiento (dr_train)
  E  OOD: pole-shift NOMINAL (N=1) por perdida (modelos n=3 del bloque A)
  F  OOD: politopo del profesor (n=2, desplazamiento POR VERTICE) por perdida
  G  Aislamiento: DR puro desde y_hat ALEATORIO vs y_hat de la red (politopo F)

Protocolo comun: base DB_ssf_RS_500_c.mat, limit=150 sistemas/(n,m,N),
split 80/20 seed=42, alpha=0.01, batch=16, lr=1e-3, dr_eval=1000.
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analisis import benchmark as bm
from analisis.validate_projection import build_operator
from analisis.benchmark_dr_vs_cvxpy import dr_checkpoints
from red.core import LMICore

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)

OUT = Path(__file__).resolve().parent / "resultados" / "reporte"
OUT.mkdir(parents=True, exist_ok=True)

SEED, LIMIT, ALPHA = 42, 150, 0.01
EPOCHS = 100            # bloques A/C/F (los baratos); B/D usan 15 (implicit/unroll-500 son lentos)
BUDGETS = [100, 250, 500, 1000, 2000, 4000, 8000]
LOSSES = ["control", "control_margen", "paper"]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"control": "#2471a3", "control_margen": "#1e8449", "paper": "#a93226",
          "aleatorio": "#7f8c8d"}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  fig -> {name}.png/.pdf")


# ============================ Bloque A: perdidas ===========================
def bloque_A():
    print("\n########## A: perdidas (n=3, N=2..5, unrolling-30, 100 epocas) ##########")
    tabla, results = bm.grid_search(
        {"loss": LOSSES}, arch="actuadores", n=3, N_list=[2, 3, 4, 5], m=1,
        dr_train=30, dr_eval=1000, epochs=EPOCHS, limit=LIMIT, seed=SEED,
        compare_cvxpy=True, cvxpy_max_systems=15)
    tabla.to_csv(OUT / "A_losses.csv", index=False)
    # detalle por N (para las barras agrupadas)
    det = pd.concat([r["df"].assign(loss=t) for r, t in zip(results, LOSSES)], ignore_index=True)
    det.to_csv(OUT / "A_losses_porN.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    Ns = sorted(det.N.unique()); x = np.arange(len(Ns)); w = .26
    for i, ls in enumerate(LOSSES):
        sub = det[det.loss == ls].set_index("N").loc[Ns]
        axs[0].bar(x + (i - 1) * w, sub.stable_pct, w, label=ls, color=COLORS[ls], alpha=.85)
    axs[0].plot(x, det[det.loss == LOSSES[0]].set_index("N").loc[Ns].cvxpy_pct,
                "k--o", ms=4, label="CVXPY factible (techo)")
    axs[0].set_xticks(x); axs[0].set_xticklabels([f"N={N}" for N in Ns])
    axs[0].set(ylabel="% estabilizados", ylim=(0, 100), title="Estabilizacion por perdida y N")
    axs[0].legend(fontsize=8)
    b = axs[1].bar(tabla.loss, tabla.pct_red_mejor_loss,
                   color=[COLORS[l] for l in tabla.loss], alpha=.85)
    axs[1].bar_label(b, fmt="%.1f", fontsize=8)
    axs[1].set(ylabel="% sistemas red mejor que CVXPY",
               title="Red vs certificado factible de CVXPY (segun cada perdida)")
    savefig(fig, "fig_A_losses")
    return results  # modelos n=3 reutilizados en E


# ========================= Bloque B: backprop ==============================
def bloque_B():
    print("\n########## B: unrolling vs implicito (n=3, 15 epocas) ##########")
    tabla, _ = bm.grid_search(
        {"backprop": ["unrolling", "implicit"]}, arch="actuadores", n=3,
        N_list=[2, 3, 4, 5], m=1, dr_train=30, dr_eval=1000, epochs=15,
        limit=LIMIT, seed=SEED, compare_cvxpy=False)
    tabla.to_csv(OUT / "B_backprop.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, col, tit in zip(axs, ["stable_pct", "t_train_s", "mem_train_mb"],
                            ["% estabilizados", "tiempo de train (s)", "memoria pico (+MB)"]):
        b = ax.bar(tabla.backprop, tabla[col], color=["#7f8c8d", "#1e8449"], alpha=.85)
        ax.bar_label(b, fmt="%.1f", fontsize=8); ax.set_title(tit)
    savefig(fig, "fig_B_backprop")


# ==================== Bloque C: escalera de arquitecturas ==================
def bloque_C():
    print("\n########## C: vanilla / vertices / actuadores (celda n=3, N=2, 100 epocas) ##########")
    runs = [
        ("vanilla (paper)", dict(arch="vanilla", backprop="implicit", dr_train=500, loss="paper")),
        ("vertices",        dict(arch="vertices", backprop="unrolling", dr_train=30, loss="control")),
        ("actuadores",      dict(arch="actuadores", backprop="unrolling", dr_train=30, loss="control")),
    ]
    rows = []
    for tag, kw in runs:
        r = bm.run_experiment(n=3, N_list=[2], m=1, dr_eval=1000, epochs=EPOCHS,
                              limit=LIMIT, seed=SEED, compare_cvxpy=True,
                              cvxpy_max_systems=15, verbose=True, **kw)
        rows.append(dict(arch=tag, loss=kw["loss"], backprop=kw["backprop"],
                         stable_pct=r["stable_pct"], decay_pct=float(r["df"].decay_pct.mean()),
                         cvxpy_pct=r["cvxpy_pct"], t_train_s=r["t_train_s"],
                         infer_ms_per_sys=r["infer_ms_per_sys"],
                         cvxpy_ms_per_sys=r["cvxpy_ms_per_sys"],
                         n_params=sum(p.numel() for p in
                                      (list(r["model"].values())[0] if isinstance(r["model"], dict)
                                       else r["model"]).parameters())))
    df = pd.DataFrame(rows); df.to_csv(OUT / "C_arquitecturas.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    b = axs[0].bar(df.arch, df.stable_pct, color=["#a93226", "#7f8c8d", "#1e8449"], alpha=.85)
    axs[0].bar_label(b, fmt="%.1f", fontsize=8)
    axs[0].axhline(df.cvxpy_pct.mean(), ls="--", color="k", alpha=.5, label="CVXPY factible")
    axs[0].set(ylabel="% estabilizados", ylim=(0, 100), title="Celda comun $n_x=3$, $N=2$")
    axs[0].legend(fontsize=8)
    axs[1].bar(df.arch, df.infer_ms_per_sys, color="#2471a3", alpha=.85, label="red (batch)")
    axs[1].axhline(df.cvxpy_ms_per_sys.mean(), ls="--", color="k", alpha=.6, label="CVXPY / sistema")
    axs[1].set_yscale("log"); axs[1].set(ylabel="ms / sistema (log)", title="Inferencia")
    axs[1].legend(fontsize=8)
    savefig(fig, "fig_C_arquitecturas")


# ======================= Bloque D: dr_train ================================
def bloque_D():
    print("\n########## D: iteraciones desenrolladas en train (15 epocas) ##########")
    tabla, _ = bm.grid_search(
        {"dr_train": [10, 30, 100, 500]}, arch="actuadores", n=3, N_list=[2, 3, 4, 5],
        m=1, dr_eval=1000, epochs=15, limit=LIMIT, seed=SEED, compare_cvxpy=False)
    tabla.to_csv(OUT / "D_drtrain.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].plot(tabla.dr_train, tabla.stable_pct, "o-", color="#2471a3")
    axs[0].set_xscale("log"); axs[0].set(xlabel="iteraciones DR en train (log)",
                                         ylabel="% estabilizados", title="Estabilizacion vs dr_train")
    axs[1].plot(tabla.dr_train, tabla.t_train_s, "s-", color="#a93226")
    axs[1].set_xscale("log"); axs[1].set(xlabel="iteraciones DR en train (log)",
                                         ylabel="tiempo de train (s)", title="Costo de entrenamiento")
    savefig(fig, "fig_D_drtrain")


# ============ Bloque E: OOD pole-shift nominal (modelos de A) ==============
def bloque_E(results_A):
    print("\n########## E: pole-shift nominal N=1 (n=3) por perdida ##########")
    A = [[0, 1, 0], [0, 0, 1], [2, -3, 1]]; B = [[0], [0], [1]]
    shifts = np.linspace(-1.0, 2.0, 13)
    rows = []
    for ls, r in zip(LOSSES, results_A):
        mdl = r["model"]
        for s in shifts:
            Ap, Bp = bm.system_to_polytope(A, B)
            Ap = bm.shift_poles(Ap, s)
            feas = bm.check_stabilizable(mdl, Ap, Bp, verbose=False)
            it = bm.iters_to_stabilize(mdl, Ap, Bp, budgets=BUDGETS)
            rows.append(dict(loss=ls, shift=s,
                             absc_ol=float(torch.linalg.eigvals(Ap[0]).real.max()),
                             lmi_factible=feas, iters_min=it["iters_min"],
                             peor_eig_max=it["worst_eig_by_iters"][max(BUDGETS)],
                             estabilizado_en_max=it["estabilizado_en_max"]))
    df = pd.DataFrame(rows); df.to_csv(OUT / "E_nominal_shift.csv", index=False)
    fig = _fig_ood(df, x="shift", xlabel="shift (A + s·I)",
                   title_pref="Nominal $N=1$, $n_x=3$")
    savefig(fig, "fig_E_nominal")
    return df


# ============ Bloque F: politopo del profesor (n=2) por perdida ============
A1 = [[0, 1], [-2, -2]]; D1 = [[2, 0], [0, 1]]
A2 = [[0, 1], [-2, -3]]; D2 = [[-2, 0], [0, 1]]
Bg = [[0], [1]]
DELTAS = np.linspace(0.0, 2.0, 15)


def bloque_F():
    print("\n########## F: politopo del profesor (n=2, D por vertice) por perdida ##########")
    modelos = {}
    for ls in LOSSES:
        r = bm.run_experiment(arch="actuadores", n=2, N_list=[2, 3], m=1, dr_train=30,
                              dr_eval=1000, backprop="unrolling", loss=ls, epochs=EPOCHS,
                              limit=LIMIT, seed=SEED, compare_cvxpy=False, verbose=True)
        modelos[ls] = r["model"]
    Ap0, Bp0 = bm.polytope_from_vertices([A1, A2], [Bg])
    rows = []
    for d in DELTAS:
        Ap = bm.shift_poles(Ap0, d, directions=[D1, D2])
        feas = bm.check_stabilizable(modelos[LOSSES[0]], Ap, Bp0, verbose=False)
        ev1 = np.linalg.eigvals(np.asarray(Ap[0])); ev2 = np.linalg.eigvals(np.asarray(Ap[1]))
        for ls in LOSSES:
            it = bm.iters_to_stabilize(modelos[ls], Ap, Bp0, budgets=BUDGETS)
            rows.append(dict(loss=ls, delta=d, lmi_factible=feas,
                             absc_A1=float(ev1.real.max()), absc_A2=float(ev2.real.max()),
                             eig_A1=" ".join(f"{e:.3f}" for e in np.round(ev1, 3)),
                             eig_A2=" ".join(f"{e:.3f}" for e in np.round(ev2, 3)),
                             iters_min=it["iters_min"],
                             peor_eig_max=it["worst_eig_by_iters"][max(BUDGETS)],
                             estabilizado_en_max=it["estabilizado_en_max"]))
    df = pd.DataFrame(rows); df.to_csv(OUT / "F_profesor.csv", index=False)
    fig = _fig_ood(df, x="delta", xlabel=r"$\delta$ ($A_i + \delta D_i$)",
                   title_pref="Politopo del profesor ($n_x=2$, $N=2$)",
                   absc_cols=("absc_A1", "absc_A2"))
    savefig(fig, "fig_F_profesor")
    return df


def _fig_ood(df, x, xlabel, title_pref, absc_cols=("absc_ol",)):
    """Figura estandar OOD: (1) abscisa lazo abierto, (2) iters_min por perdida,
    (3) peor autovalor CL al presupuesto maximo por perdida."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    sub0 = df[df.loss == df.loss.iloc[0]]
    for col, mk in zip(absc_cols, "os"):
        axs[0].plot(sub0[x], sub0[col], marker=mk, ms=4,
                    label=col.replace("absc_", "vertice ").replace("ol", "nominal"))
    axs[0].axhline(0, ls="--", color="k", alpha=.5)
    axs[0].set(xlabel=xlabel, ylabel=r"$\max\,\mathrm{Re}\,\lambda$ (lazo abierto)",
               title=f"{title_pref}: abscisa")
    axs[0].legend(fontsize=8)
    for ls in df.loss.unique():
        sub = df[df.loss == ls]
        ok = sub.iters_min.notna()
        axs[1].plot(sub.loc[ok, x], sub.loc[ok, "iters_min"], "o-", ms=4,
                    color=COLORS[ls], label=ls)
        if (~ok).any():
            axs[1].scatter(sub.loc[~ok, x], [max(BUDGETS) * 1.6] * int((~ok).sum()),
                           marker="x", color=COLORS[ls], s=45)
        axs[2].plot(sub[x], sub.peor_eig_max, "o-", ms=4, color=COLORS[ls], label=ls)
    axs[1].axhline(max(BUDGETS), ls="--", color="k", alpha=.4, lw=1)
    axs[1].set_yscale("log")
    axs[1].set(xlabel=xlabel, ylabel="iteraciones DR necesarias (log)",
               title="Costo de iteraciones (x = no logrado)")
    axs[1].legend(fontsize=8)
    axs[2].axhline(0, ls="--", color="k", alpha=.5)
    axs[2].set(xlabel=xlabel, ylabel=f"peor autovalor CL @ {max(BUDGETS)} iters",
               title="Lazo cerrado al presupuesto maximo")
    axs[2].legend(fontsize=8)
    return fig


# ====== Bloque G: aislamiento — DR puro (y_hat aleatorio) vs la red ========
def bloque_G(df_F):
    print("\n########## G: DR puro desde y_hat aleatorio (politopo del profesor) ##########")
    model = LMICore(n=2, m=1, N=2, alpha=ALPHA, epsilon=1e-5, dr_iters=1, sigma=0.01).double()
    rng = np.random.default_rng(0)
    Ap0, Bp0 = bm.polytope_from_vertices([A1, A2], [Bg])
    rows = []
    for d in DELTAS:
        Ap = bm.shift_poles(Ap0, d, directions=[D1, D2])
        An, Bn = bm.normalize_system(Ap, Bp0)          # misma normalizacion que la red
        A_b, B_b = An.unsqueeze(0), Bn.unsqueeze(0)
        mins = []
        for _ in range(5):                              # 5 warm starts aleatorios
            yh = torch.tensor(rng.standard_normal((1, model.dim_y)) * 0.3)
            L, c, Minv = build_operator(model, A_b, B_b)
            ck = dr_checkpoints(model, yh, L, c, Minv, BUDGETS, 0.01)
            got = np.inf
            for it in BUDGETS:
                y, _ = ck[it]
                Q, Y = model._y_to_matrices(y)
                K = Y[0] @ torch.linalg.inv(Q[0])
                we = max(torch.linalg.eigvals(A_b[0, i] + B_b[0, i] @ K).real.max().item()
                         for i in range(2))
                if we < 0:
                    got = it; break
            mins.append(got)
        rows.append(dict(delta=d, iters_mediana=float(np.median(mins)),
                         iters_peor=float(np.max(mins))))
    df = pd.DataFrame(rows); df.to_csv(OUT / "G_dr_puro.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(df.delta, df.iters_mediana, "o-", color=COLORS["aleatorio"],
            label="DR puro, $\\hat{y}$ aleatorio (mediana de 5)")
    for ls in LOSSES:
        sub = df_F[df_F.loss == ls]
        ok = sub.iters_min.notna()
        ax.plot(sub.loc[ok, "delta"], sub.loc[ok, "iters_min"], "s-", ms=4,
                color=COLORS[ls], label=f"$\\hat{{y}}$ de la red ({ls})")
        if (~ok).any():
            ax.scatter(sub.loc[~ok, "delta"], [max(BUDGETS) * 1.6] * int((~ok).sum()),
                       marker="x", color=COLORS[ls], s=45)
    ax.axhline(max(BUDGETS), ls="--", color="k", alpha=.4, lw=1)
    ax.set_yscale("log")
    ax.set(xlabel=r"$\delta$", ylabel="iteraciones DR para estabilizar (log)",
           title="El solver no es el cuello de botella: $\\hat{y}$ decide (x = no logrado)")
    ax.legend(fontsize=8)
    savefig(fig, "fig_G_aislamiento")


def main():
    t0 = time.perf_counter()
    resA = bloque_A()
    bloque_B()
    bloque_C()
    bloque_D()
    bloque_E(resA)
    dfF = bloque_F()
    bloque_G(dfF)
    meta = dict(seed=SEED, limit=LIMIT, alpha=ALPHA, epochs=EPOCHS, budgets=BUDGETS,
                losses=LOSSES, deltas=list(map(float, DELTAS)),
                total_s=round(time.perf_counter() - t0, 1))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nTODO LISTO en {meta['total_s']:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
