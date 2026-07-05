"""
experimento_trayectoria.py
==========================
Anatomia del estancamiento: trayectoria del PEOR AUTOVALOR de lazo cerrado
(max_i Re lambda(A_i + B_i K), K = Y Q^{-1}) a lo largo de las iteraciones de
Douglas-Rachford, muestreada densamente (cada 20 iters hasta 8000), sobre el
politopo del profesor. Tres paneles:

  1. ESTANCAMIENTO (red `control`): delta facil converge; delta medio baja
     lento; delta extremo se queda en meseta POSITIVA (el fallo).
  2. CRUCE TRANSITORIO / zig-zag (red `paper`): la trayectoria NO es monotona;
     cruza la region estable de pasada y vuelve a salir (por que iters_min
     solo vale junto a estabilizado_en_max).
  3. EL REMEDIO (delta=2): mismo sistema, tres y_hat — control (estancado),
     control_margen (converge), aleatorio (converge rapido). El solver no es
     el cuello de botella: y_hat decide.

Salidas en analisis/resultados/reporte/: trayectoria_dr.csv, fig_trayectoria_dr.*
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analisis import benchmark as bm
from analisis.benchmark_dr_vs_cvxpy import dr_checkpoints

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)

OUT = Path(__file__).resolve().parent / "resultados" / "reporte"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})

A1 = [[0, 1], [-2, -2]]; D1 = [[2, 0], [0, 1]]
A2 = [[0, 1], [-2, -3]]; D2 = [[-2, 0], [0, 1]]
MILESTONES = list(range(20, 8001, 20))          # cada 20 iters -> 400 puntos


def trayectoria(model, Ap, Bp, y_hat=None):
    """Peor autovalor de lazo cerrado en cada milestone de UNA pasada de DR.
    y_hat=None usa el del encoder; si no, el tensor dado (1, dim_y)."""
    A, B = bm.normalize_system(torch.as_tensor(Ap).double(), torch.as_tensor(Bp).double())
    Ab, Bb = A.unsqueeze(0), B.unsqueeze(0)
    old_it, old_impl = model.dr_iters, model.use_implicit
    model.use_implicit = False; model.eval()
    with torch.no_grad():
        if y_hat is None:
            y_hat = model(Ab, Bb, return_unconstrained=True)
        L, c, M_inv = model._dr_precompute(Ab, Bb)
        ck = dr_checkpoints(model, y_hat, L, c, M_inv, MILESTONES, model.sigma)
        out = []
        for it in MILESTONES:
            Q, Y = model._y_to_matrices(ck[it][0])
            K = Y[0] @ torch.linalg.inv(Q[0])
            out.append(max(torch.linalg.eigvals(Ab[0, i] + Bb[0, i] @ K).real.max().item()
                           for i in range(Ab.shape[1])))
    model.dr_iters, model.use_implicit = old_it, old_impl
    return np.array(out)


def main():
    t0 = time.perf_counter()
    modelos = {}
    for ls in ["control", "control_margen", "paper"]:
        r = bm.run_experiment(arch="actuadores", n=2, N_list=[2, 3], m=1, dr_train=30,
                              dr_eval=1000, loss=ls, epochs=100, limit=150, seed=42,
                              compare_cvxpy=False, verbose=True)
        modelos[ls] = r["model"]

    Ap0, Bp0 = bm.polytope_from_vertices([A1, A2], [[[0], [1]]])
    politopo = lambda d: bm.shift_poles(Ap0, d, directions=[D1, D2])

    rng = np.random.default_rng(0)
    filas = []

    def curva(tag, model, d, y_hat=None):
        tr = trayectoria(model, politopo(d), Bp0, y_hat)
        for it, we in zip(MILESTONES, tr):
            filas.append(dict(serie=tag, delta=d, iter=it, peor_eig=we))
        return tr

    # Panel 1: estancamiento (control)
    p1 = {d: curva(f"control δ={d:g}", modelos["control"], d) for d in [0.43, 1.29, 2.0]}
    # Panel 2: zig-zag / cruce transitorio (paper)
    p2 = {d: curva(f"paper δ={d:g}", modelos["paper"], d) for d in [1.29, 1.57]}
    # Panel 3: remedio en δ=2 — mismo sistema, tres y_hat distintos
    p3 = {"control": p1[2.0]}
    p3["control_margen"] = curva("margen δ=2", modelos["control_margen"], 2.0)
    with torch.no_grad():
        A, B = bm.normalize_system(politopo(2.0), Bp0)
        dim = modelos["control"](A.unsqueeze(0), B.unsqueeze(0), return_unconstrained=True).shape[-1]
    y_ale = torch.tensor(rng.standard_normal((1, dim)) * 0.3)
    p3["aleatorio"] = curva("aleatorio δ=2", modelos["control"], 2.0, y_hat=y_ale)

    pd.DataFrame(filas).to_csv(OUT / "trayectoria_dr.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    YLIM = (-2.1, 1.6)          # los picos (hasta +5e4, K explota con Q casi singular) se recortan
    az = {0.43: "#7fb3d5", 1.29: "#2471a3", 2.0: "#154360"}
    for d, tr in p1.items():
        axs[0].plot(MILESTONES, tr, color=az[d], lw=1.4, label=f"$\\delta$={d:g}")
    axs[0].set_title("1. Estancamiento (red `control`)")
    axs[0].annotate("$\\delta$=2: meseta POSITIVA,\nnunca cruza", xy=(2400, 1.05),
                    fontsize=8, color="#154360")
    rj = {1.29: "#e6867a", 1.57: "#a93226"}
    for d, tr in p2.items():
        axs[1].plot(MILESTONES, tr, color=rj[d], lw=1.4, label=f"$\\delta$={d:g}")
        axs[1].fill_between(MILESTONES, 0, np.minimum(tr, 0), color=rj[d], alpha=.25)
    axs[1].set_title("2. Cruce transitorio / zig-zag (red `paper`)")
    axs[1].annotate("cruza de pasada…\ny se devuelve", xy=(70, -1.3), fontsize=8, color="#a93226")
    for tag, c in [("control", "#154360"), ("control_margen", "#1e8449"), ("aleatorio", "#7f8c8d")]:
        axs[2].plot(MILESTONES, p3[tag], color=c, lw=1.4,
                    label={"control": "$\\hat{y}$ red (control)",
                           "control_margen": "$\\hat{y}$ red (control_margen)",
                           "aleatorio": "$\\hat{y}$ aleatorio"}[tag])
    axs[2].set_title("3. Mismo sistema ($\\delta$=2): $\\hat{y}$ decide")
    for ax in axs:
        ax.axhline(0, ls="--", color="k", alpha=.6, lw=1)
        ax.set_xscale("log"); ax.set_ylim(*YLIM)
        ax.set_xlabel("iteración de Douglas-Rachford (log)")
        ax.legend(fontsize=8, loc="lower left")
    axs[0].set_ylabel(r"peor autovalor de lazo cerrado  $\max_i \mathrm{Re}\,\lambda$")
    fig.text(0.995, 0.02, "picos tempranos recortados (hasta $+5\\cdot10^4$): K=YQ$^{-1}$ explota cuando Q pasa por casi-singular",
             ha="right", fontsize=7.5, style="italic", color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "fig_trayectoria_dr.png", dpi=150)
    fig.savefig(OUT / "fig_trayectoria_dr.pdf")
    print(f"LISTO en {time.perf_counter() - t0:.0f}s -> fig_trayectoria_dr.png/.pdf")


if __name__ == "__main__":
    main()
