"""
experimento_hparams.py
======================
Barrido AMPLIADO de hiperparametros alrededor de la config base del reporte,
para ver si los hallazgos de experimento_reporte.py cambian o se repiten.

Diseno: UN eje a la vez sobre la base (atribucion limpia), mas la interaccion
dr_train x epocas (¿mas epocas rescatan al unrolling largo?), el eje dr_eval
sin reentrenar (mismo modelo, distintos presupuestos de inferencia hasta 5000)
y un re-chequeo OOD (politopo del profesor) con 4x mas epocas.

BASE: actuadores, n=3, N=[2,3,4,5], m=1, unrolling, loss=control,
      dr_train=30, dr_eval=1000, epochs=100, batch=16, lr=1e-3, alpha=0.01,
      limit=150, seed=42.

Ejes:  lr [3e-4, 1e-3, 3e-3, 1e-2] | epochs [15, 50, 100, 200, 400]
       batch [8, 16, 32, 64]       | dr_train [5, 30, 120, 500] x epochs [15, 100]
       dr_eval [100..5000] (sin reentrenar) | alpha [0.001, 0.01, 0.05, 0.1]
       orden n [2, 3, 4, 5]

Salidas en analisis/resultados/reporte/: H_*.csv y fig_H_*.png/pdf
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analisis import benchmark as bm
from entrenamiento.training import load_vertices, split_items

torch.set_default_dtype(torch.float64)
torch.set_num_threads(6)

OUT = Path(__file__).resolve().parent / "resultados" / "reporte"
OUT.mkdir(parents=True, exist_ok=True)

BASE = dict(arch="actuadores", n=3, N_list=[2, 3, 4, 5], m=1, dr_train=30,
            dr_eval=1000, backprop="unrolling", loss="control", epochs=100,
            batch=16, lr=1e-3, alpha=0.01, limit=150, seed=42,
            compare_cvxpy=False, verbose=True)
BUDGETS = [100, 250, 500, 1000, 2000, 4000, 8000]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  fig -> {name}.png/.pdf")


def corre(**over):
    """run_experiment con la base + overrides; devuelve fila resumen + resultado."""
    kw = {**BASE, **over}
    r = bm.run_experiment(**kw)
    row = dict(stable_pct=r["stable_pct"], decay_pct=float(r["df"].decay_pct.mean()),
               t_train_s=r["t_train_s"], loss_final=r["loss_hist"][-1],
               worst_mean=float(r["df"].worst_mean.mean()))
    if r["cvxpy_pct"] is not None:
        row["cvxpy_pct"] = r["cvxpy_pct"]
    return row, r


def eje(nombre, valores, base_val, base_row, fila_extra=None, **kw_extra):
    """Barre `nombre` sobre `valores`; reutiliza la corrida base para base_val."""
    rows = []
    for v in valores:
        if v == base_val and not kw_extra:
            rows.append({nombre: v, **base_row, "es_base": True})
            continue
        print(f"\n=== {nombre} = {v} ===")
        row, r = corre(**{nombre: v}, **kw_extra)
        rows.append({nombre: v, **row, "es_base": False})
        if fila_extra:
            fila_extra(v, r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"H_{nombre}.csv", index=False)
    print(df.round(3).to_string(index=False))
    return df


def linea(df, x, titulo, xlabel, logx=True, extra_y=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].plot(df[x], df.stable_pct, "o-", color="#2471a3")
    if extra_y is not None and extra_y in df:
        axs[0].plot(df[x], df[extra_y], "k--o", ms=4, label="techo CVXPY")
        axs[0].legend(fontsize=8)
    if logx: axs[0].set_xscale("log")
    axs[0].set(xlabel=xlabel, ylabel="% estabilizados", title=titulo)
    axs[1].plot(df[x], df.t_train_s, "s-", color="#a93226")
    if logx: axs[1].set_xscale("log")
    axs[1].set(xlabel=xlabel, ylabel="tiempo de train (s)", title="Costo")
    return fig


def main():
    t0 = time.perf_counter()

    print("\n########## BASE ##########")
    base_row, base_res = corre()
    print(pd.Series(base_row).round(3).to_string())

    # ---------- dr_eval: SIN reentrenar (mismo modelo base) ----------
    print("\n########## dr_eval (mismo modelo, 100..5000 iters de inferencia) ##########")
    data = {N: split_items(load_vertices(3, 1, N, limit=BASE["limit"]), seed=BASE["seed"])[1]
            for N in BASE["N_list"]}
    rows = []
    for de in [100, 250, 500, 1000, 2000, 5000]:
        per_N = [bm._eval_stab(base_res["model"], data[N], de, BASE["alpha"], BASE["batch"])
                 for N in BASE["N_list"]]
        rows.append(dict(dr_eval=de,
                         stable_pct=float(np.mean([p["stable_pct"] for p in per_N])),
                         decay_pct=float(np.mean([p["decay_pct"] for p in per_N])),
                         infer_ms_per_sys=float(np.mean([p["infer_ms_per_sys"] for p in per_N]))))
        print(f"  dr_eval={de:>5}  estabiliza={rows[-1]['stable_pct']:.1f}%  "
              f"infer={rows[-1]['infer_ms_per_sys']:.2f} ms/sys")
    dfe = pd.DataFrame(rows); dfe.to_csv(OUT / "H_dr_eval.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].plot(dfe.dr_eval, dfe.stable_pct, "o-", color="#2471a3", label="% estabilizados")
    axs[0].plot(dfe.dr_eval, dfe.decay_pct, "s-", color="#1e8449", label="% cumple decay α")
    axs[0].set_xscale("log"); axs[0].legend(fontsize=8)
    axs[0].set(xlabel="iteraciones DR en inferencia (log)", ylabel="%",
               title="Mismo modelo, mas presupuesto de inferencia")
    axs[1].plot(dfe.dr_eval, dfe.infer_ms_per_sys, "s-", color="#a93226")
    axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[1].set(xlabel="iteraciones DR en inferencia (log)", ylabel="ms/sistema (log)",
               title="Costo de inferencia")
    savefig(fig, "fig_H_dr_eval")

    # ---------------------------- lr ----------------------------
    print("\n########## lr ##########")
    df = eje("lr", [3e-4, 1e-3, 3e-3, 1e-2], 1e-3, base_row)
    savefig(linea(df, "lr", "Learning rate", "lr (log)"), "fig_H_lr")

    # -------------------------- epochs --------------------------
    print("\n########## epochs ##########")
    df = eje("epochs", [15, 50, 100, 200, 400], 100, base_row)
    savefig(linea(df, "epochs", "Epocas de entrenamiento", "epocas", logx=False),
            "fig_H_epochs")

    # --------------------------- batch --------------------------
    print("\n########## batch ##########")
    df = eje("batch", [8, 16, 32, 64], 16, base_row)
    savefig(linea(df, "batch", "Tamano de batch", "batch (log)"), "fig_H_batch")

    # ----------------- dr_train x epochs (interaccion) ----------------
    print("\n########## dr_train x epochs ##########")
    rows = []
    for ep in [15, 100]:
        for dt in [5, 30, 120, 500]:
            if dt == 30 and ep == 100:
                rows.append(dict(dr_train=dt, epochs=ep, **base_row)); continue
            print(f"\n=== dr_train={dt}, epochs={ep} ===")
            row, _ = corre(dr_train=dt, epochs=ep)
            rows.append(dict(dr_train=dt, epochs=ep, **row))
    dfi = pd.DataFrame(rows); dfi.to_csv(OUT / "H_drtrain_epochs.csv", index=False)
    print(dfi.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for ep, c in [(15, "#7f8c8d"), (100, "#2471a3")]:
        sub = dfi[dfi.epochs == ep]
        ax.plot(sub.dr_train, sub.stable_pct, "o-", color=c, label=f"{ep} epocas")
    ax.set_xscale("log")
    ax.set(xlabel="iteraciones DR en train (log)", ylabel="% estabilizados",
           title="¿Mas epocas rescatan al unrolling largo?")
    ax.legend(fontsize=8)
    savefig(fig, "fig_H_drtrain_epochs")

    # --------------------------- alpha --------------------------
    print("\n########## alpha (con techo CVXPY por alpha) ##########")
    df = eje("alpha", [0.001, 0.01, 0.05, 0.1], None, base_row,
             compare_cvxpy=True, cvxpy_max_systems=10)
    savefig(linea(df, "alpha", "Decay-rate objetivo α", "α (log)", extra_y="cvxpy_pct"),
            "fig_H_alpha")

    # ------------------------- orden n --------------------------
    print("\n########## orden n (con techo CVXPY por orden) ##########")
    df = eje("n", [2, 3, 4, 5], None, base_row,
             compare_cvxpy=True, cvxpy_max_systems=10)
    savefig(linea(df, "n", "Orden del sistema $n_x$", "$n_x$", logx=False,
                  extra_y="cvxpy_pct"), "fig_H_orden")

    # ------- re-chequeo OOD: politopo del profesor con 4x epocas -------
    print("\n########## OOD profesor: epochs 100 vs 400 (control / control_margen) ##########")
    A1 = [[0, 1], [-2, -2]]; D1 = [[2, 0], [0, 1]]
    A2 = [[0, 1], [-2, -3]]; D2 = [[-2, 0], [0, 1]]
    Ap0, Bp0 = bm.polytope_from_vertices([A1, A2], [[[0], [1]]])
    rows = []
    for ls in ["control", "control_margen"]:
        for ep in [100, 400]:
            print(f"\n=== OOD {ls}, {ep} epocas ===")
            r = bm.run_experiment(**{**BASE, "n": 2, "N_list": [2, 3], "loss": ls,
                                     "epochs": ep})
            for d in np.linspace(0.0, 2.0, 15):
                Ap = bm.shift_poles(Ap0, d, directions=[D1, D2])
                it = bm.iters_to_stabilize(r["model"], Ap, Bp0, budgets=BUDGETS)
                rows.append(dict(loss=ls, epochs=ep, delta=d,
                                 stable_pct_indist=r["stable_pct"],
                                 iters_min=it["iters_min"],
                                 peor_eig_max=it["worst_eig_by_iters"][max(BUDGETS)],
                                 estabilizado_en_max=it["estabilizado_en_max"]))
    dfo = pd.DataFrame(rows); dfo.to_csv(OUT / "H_ood_epochs.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    estilos = {(ls, ep): (c, m) for (ls, c) in [("control", "#2471a3"), ("control_margen", "#1e8449")]
               for (ep, m) in [(100, "--"), (400, "-")]}
    for (ls, ep), (c, m) in estilos.items():
        sub = dfo[(dfo.loss == ls) & (dfo.epochs == ep)]
        ok = sub.iters_min.notna()
        axs[0].plot(sub.loc[ok, "delta"], sub.loc[ok, "iters_min"], m, marker="o", ms=3,
                    color=c, label=f"{ls}, {ep} ep")
        if (~ok).any():
            axs[0].scatter(sub.loc[~ok, "delta"], [max(BUDGETS) * 1.6] * int((~ok).sum()),
                           marker="x", color=c, s=40)
        axs[1].plot(sub.delta, sub.peor_eig_max, m, marker="o", ms=3, color=c,
                    label=f"{ls}, {ep} ep")
    axs[0].axhline(max(BUDGETS), ls=":", color="k", alpha=.4, lw=1)
    axs[0].set_yscale("log"); axs[0].legend(fontsize=7)
    axs[0].set(xlabel=r"$\delta$", ylabel="iteraciones DR necesarias (log)",
               title="OOD profesor: ¿mas epocas cambian el hallazgo? (x = no logrado)")
    axs[1].axhline(0, ls="--", color="k", alpha=.5); axs[1].legend(fontsize=7)
    axs[1].set(xlabel=r"$\delta$", ylabel=f"peor autovalor CL @ {max(BUDGETS)}",
               title="Lazo cerrado al presupuesto maximo")
    savefig(fig, "fig_H_ood_epochs")

    print(f"\nTODO LISTO en {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
