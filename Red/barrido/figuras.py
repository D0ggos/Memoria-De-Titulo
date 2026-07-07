"""
figuras.py  —  figuras del barrido y del OOD (Parte F)
======================================================
Lee los CSV/parquet CRUDOS del barrido (Parte D) y del OOD (Parte E) y produce las
figuras. NO recomputa nada: solo agrega (aqui SI se puede promediar; la regla de "no
promediar" es para los CSV crudos, no para el analisis). Robusto a datos faltantes:
si un shard no existe, avisa y salta esa figura.

Estilo consistente con los scripts del repo: serif, sin spines top/right, grid alpha 0.25.

Uso:
  python -m barrido.figuras --barrido analisis/resultados/barrido/E1 \
                            --ood analisis/resultados/barrido/OOD \
                            --out analisis/resultados/barrido/figuras
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
import matplotlib.pyplot as plt                        # noqa: E402
# NB: no forzamos backend al importar (romperia los plots inline del notebook). El CLI
# (main) cambia a 'pdf' headless; las funciones usan fig.savefig, que anda en cualquier backend.

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})

LOSS_ORDER = ["paper", "control", "control_margen", "esfuerzo", "condicionamiento", "margen_norm"]
_CMAP = plt.cm.tab10


# --------------------------- IO robusto ------------------------------------
def read_many(folder):
    """Concatena todos los shards (*.parquet o *.csv) de una carpeta. DataFrame vacio si no hay."""
    folder = Path(folder)
    if not folder.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(list(folder.glob("*.parquet")) + list(folder.glob("*.csv"))):
        try:
            frames.append(pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))
        except Exception as e:                          # noqa: BLE001
            print(f"  aviso: no pude leer {p.name}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _save(fig, out, name):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {name}.pdf/.png")


def _loss_color(loss):
    i = LOSS_ORDER.index(loss) if loss in LOSS_ORDER else 7
    return _CMAP(i)


def _final_epoch(df):
    return df.epoch.max() if "epoch" in df else None


# ============================ figuras de entrenamiento =====================
def fig_curvas_loss(barrido, out):
    df = read_many(Path(barrido) / "loss")
    if df.empty:
        print("  (sin shards de loss)"); return
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for loss, g in df.groupby("loss"):
        curve = g.groupby("epoch").train_loss.mean()
        ax.plot(curve.index, curve.values, label=loss, color=_loss_color(loss))
    ax.set(xlabel="época", ylabel="loss de entrenamiento (media sobre configs)",
           title="Curvas de pérdida por función de pérdida")
    ax.legend(fontsize=8, ncol=2)
    _save(fig, out, "train_curvas_loss")


def fig_estabilidad(barrido, out):
    df = read_many(Path(barrido) / "results")
    if df.empty or "stable" not in df:
        print("  (sin resultados de estabilidad)"); return
    ep = _final_epoch(df)
    for ring in sorted(df.ring.unique()):
        sub = df[(df.ring == ring) & (df.epoch == ep)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for loss, g in sub.groupby("loss"):
            agg = g.groupby("dr_eval").stable.mean() * 100
            ax.plot(agg.index, agg.values, marker="o", label=loss, color=_loss_color(loss))
        ax.set(xscale="log", xlabel="presupuesto DR (dr_eval)", ylabel="% estabilizado",
               ylim=(0, 101), title=f"Estabilización vs dr_eval — anillo {ring} (época {ep})")
        ax.legend(fontsize=8, ncol=2)
        _save(fig, out, f"train_estabilidad_{ring}")


def fig_inferencia_tiempo(barrido, out):
    df = read_many(Path(barrido) / "results")
    if df.empty or "t_unit_ms" not in df:
        print("  (sin tiempos de inferencia)"); return
    ep = _final_epoch(df)
    sub = df[df.epoch == ep]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    agg = sub.groupby("dr_eval").t_unit_ms.mean()
    ax.plot(agg.index, agg.values, marker="o", color="#1e8449", label="red (DR)")
    ax.set(xscale="log", yscale="log", xlabel="presupuesto DR (dr_eval)",
           ylabel="tiempo de inferencia por sistema [ms]",
           title="Tiempo de inferencia unitario vs presupuesto DR")
    ax.legend(fontsize=8)
    _save(fig, out, "train_inferencia_tiempo")


def fig_meta_tiempo_memoria(barrido, out):
    df = read_many(Path(barrido) / "meta")
    if df.empty:
        print("  (sin meta de tiempo/memoria)"); return
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, t in ((axs[0], "t_train_s", "tiempo de entrenamiento [s]"),
                       (axs[1], "mem_peak_mb", "memoria pico [MB]")):
        if col not in df:
            continue
        agg = df.groupby("loss")[col].mean().reindex([l for l in LOSS_ORDER if l in df.loss.unique()])
        b = ax.bar(agg.index, agg.values, color=[_loss_color(l) for l in agg.index], alpha=.85)
        ax.bar_label(b, fmt="%.0f", fontsize=7)
        ax.set_title(t); ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Costo de entrenamiento por pérdida (media sobre configs)", y=1.03)
    fig.tight_layout()
    _save(fig, out, "train_tiempo_memoria")


def fig_cvxpy_comparacion(barrido, out):
    df = read_many(Path(barrido) / "results")
    if df.empty or "cvxpy_feasible" not in df:
        print("  (sin techo CVXPY; corre el barrido con --cvxpy)"); return
    ep = _final_epoch(df)
    dr = df.dr_eval.max()
    sub = df[(df.epoch == ep) & (df.dr_eval == dr)]
    fig, ax = plt.subplots(figsize=(7, 4))
    rings = sorted(sub.ring.unique()); x = np.arange(len(rings)); w = 0.38
    red = [sub[sub.ring == r].stable.mean() * 100 for r in rings]
    cvx = [np.nanmean(sub[sub.ring == r].cvxpy_feasible) * 100 for r in rings]
    b1 = ax.bar(x - w / 2, red, w, label="red (estabiliza)", color="#1e8449", alpha=.85)
    b2 = ax.bar(x + w / 2, cvx, w, label="CVXPY (LMI factible)", color="#a93226", alpha=.85)
    ax.bar_label(b1, fmt="%.0f", fontsize=7); ax.bar_label(b2, fmt="%.0f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(rings)
    ax.set(ylabel="%", ylim=(0, 101), title=f"Red vs techo CVXPY (época {ep}, dr_eval={dr})")
    ax.legend(fontsize=8)
    _save(fig, out, "train_cvxpy_A1_A2")


# ============================ figuras del profesor =========================
def _random_iters_min(trace):
    """iters_min del control DR-puro (ŷ aleatorio): por δ, menor k con mediana worst<0."""
    r = trace[trace.source == "aleatorio"]
    out = {}
    for d, g in r.groupby("delta"):
        gg = g.groupby("k").worst.median()
        stab = gg[gg < 0]
        out[d] = stab.index.min() if len(stab) else np.nan
    return pd.Series(out)


def figP1_itersmin_vs_delta(ood, out):
    summ = read_many(Path(ood) / "mecanismo_summary")
    trace = read_many(Path(ood) / "mecanismo_trace")
    if summ.empty:
        print("  (sin mecanismo_summary; P1 omitido)"); return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for loss, g in summ.groupby("loss"):
        agg = g.groupby("delta").iters_min.median()
        ax.plot(agg.index, agg.values, marker="o", label=f"red: {loss}", color=_loss_color(loss))
    if not trace.empty:
        rim = _random_iters_min(trace)
        ax.plot(rim.index, rim.values, marker="s", ls="--", color="k", label="DR-puro (ŷ aleatorio)")
    # marca de factibilidad CVXPY: δ donde CVXPY dice infactible
    if "cvxpy_feasible" in summ:
        infeas = summ[summ.cvxpy_feasible == False].delta.unique()   # noqa: E712
        for d in infeas:
            ax.axvline(d, color="#a93226", alpha=.15, lw=6)
    ax.set(yscale="log", xlabel=r"$\delta$", ylabel="iters_min (menor DR que estabiliza)",
           title="P1 · iters_min vs δ por pérdida (banda roja = CVXPY infactible)")
    ax.legend(fontsize=7.5, ncol=2)
    _save(fig, out, "prof_P1_itersmin_vs_delta")


def figP2_trazas(ood, out, deltas=(1.0, 1.5, 2.0)):
    trace = read_many(Path(ood) / "mecanismo_trace")
    if trace.empty:
        print("  (sin mecanismo_trace; P2 omitido)"); return
    avail = np.array(sorted(trace.delta.unique()))
    sel = [avail[np.abs(avail - d).argmin()] for d in deltas]
    fig, axs = plt.subplots(2, len(sel), figsize=(4.2 * len(sel), 7), sharex=True)
    axs = np.atleast_2d(axs)
    for j, d in enumerate(sel):
        sub = trace[np.isclose(trace.delta, d)]
        for src, col in (("red", "#a93226"), ("aleatorio", "#2471a3")):
            s = sub[sub.source == src]
            w = s.groupby("k").worst.median(); r = s.groupby("k").residual.median()
            axs[0, j].plot(w.index, w.values, color=col, label=f"ŷ {src}")
            axs[1, j].plot(r.index, r.values, color=col, label=f"ŷ {src}")
        axs[0, j].axhline(0, ls="--", color="k", alpha=.5)
        axs[0, j].set_title(fr"$\delta={d:.2f}$")
        axs[1, j].set_yscale("log")
        axs[1, j].set_xlabel("iteración k de DR")
    axs[0, 0].set_ylabel(r"peor Re$\,\lambda$ de lazo cerrado")
    axs[1, 0].set_ylabel("residual de DR (log)")
    axs[0, 0].legend(fontsize=7.5); axs[1, 0].legend(fontsize=7.5)
    for ax in axs.ravel():
        ax.set_xscale("log")
    fig.suptitle("P2 · peor autovalor y residual vs k (ŷ red vs aleatorio): "
                 "el aplanamiento del residual solo aparece con ŷ de la red", y=1.02)
    fig.tight_layout()
    _save(fig, out, "prof_P2_trazas")


def figP3_certificado_vs_delta(ood, out):
    summ = read_many(Path(ood) / "mecanismo_summary")
    if summ.empty:
        print("  (sin mecanismo_summary; P3 omitido)"); return
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [("lam_min_Q_final", r"$\lambda_{\min}(Q^*)$", False),
              ("kappa_Q_final", r"$\kappa(Q^*)$", True),
              ("worst_final", r"peor Re$\,\lambda$ final", False)]
    for ax, (col, t, logy) in zip(axs, panels):
        for loss, g in summ.groupby("loss"):
            agg = g.groupby("delta")[col].median()
            ax.plot(agg.index, agg.values, marker="o", label=loss, color=_loss_color(loss))
        if logy:
            ax.set_yscale("log")
        if col == "worst_final":
            ax.axhline(0, ls="--", color="k", alpha=.5)
        ax.set(xlabel=r"$\delta$", title=t)
    axs[0].legend(fontsize=7, ncol=2)
    fig.suptitle("P3 · certificado final vs δ por pérdida", y=1.03)
    fig.tight_layout()
    _save(fig, out, "prof_P3_certificado_vs_delta")


def figP4_tiempo_vs_delta(ood, out):
    summ = read_many(Path(ood) / "mecanismo_summary")
    if summ.empty or "t_stab_red_ms" not in summ:
        print("  (sin tiempos en mecanismo_summary; P4 omitido)"); return
    fig, ax = plt.subplots(figsize=(7, 4))
    red = summ.groupby("delta").t_stab_red_ms.median()
    ax.plot(red.index, red.values, marker="o", color="#1e8449", label="red (hasta estabilizar)")
    if "t_cvxpy_ms" in summ:
        cvx = summ.groupby("delta").t_cvxpy_ms.median()
        ax.plot(cvx.index, cvx.values, marker="s", color="#a93226", label="CVXPY")
    ax.set(yscale="log", xlabel=r"$\delta$", ylabel="tiempo [ms] (log)",
           title="P4 · tiempo hasta estabilizar: red vs CVXPY")
    ax.legend(fontsize=8)
    _save(fig, out, "prof_P4_tiempo_vs_delta")


def figP5_heatmap(ood, out):
    summ = read_many(Path(ood) / "mecanismo_summary")
    if summ.empty:
        print("  (sin mecanismo_summary; P5 omitido)"); return
    piv = summ.groupby(["loss", "delta"]).iters_min.median().unstack("delta")
    piv = piv.reindex([l for l in LOSS_ORDER if l in piv.index])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    data = piv.values.astype(float)
    im = ax.imshow(np.log10(np.where(np.isnan(data), np.nan, data)), aspect="auto",
                   cmap="viridis_r", origin="lower")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([f"{d:.1f}" for d in piv.columns], rotation=90, fontsize=6)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    # marca de "falla" (iters_min NaN) con una X
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                ax.text(j, i, "×", ha="center", va="center", color="#a93226", fontsize=8)
    cb = fig.colorbar(im, ax=ax); cb.set_label(r"$\log_{10}$ iters_min")
    ax.set(xlabel=r"$\delta$", title="P5 · mapa iters_min (× = no estabiliza en el ladder)")
    fig.tight_layout()
    _save(fig, out, "prof_P5_heatmap")


# ============================ main =========================================
def main():
    ap = argparse.ArgumentParser(description="Figuras del barrido y OOD (lee CSV/parquet).")
    ap.add_argument("--barrido", default=None, help="carpeta de una etapa (con results/ loss/ meta/)")
    ap.add_argument("--ood", default=None, help="carpeta OOD (con mecanismo_summary/ trace/ ladder/)")
    ap.add_argument("--out", default=str(ROOT / "analisis" / "resultados" / "barrido" / "figuras"))
    args = ap.parse_args()
    plt.switch_backend("pdf")                          # headless para el CLI

    if args.barrido:
        print(f"Figuras de entrenamiento desde {args.barrido}")
        fig_curvas_loss(args.barrido, args.out)
        fig_estabilidad(args.barrido, args.out)
        fig_inferencia_tiempo(args.barrido, args.out)
        fig_meta_tiempo_memoria(args.barrido, args.out)
        fig_cvxpy_comparacion(args.barrido, args.out)
    if args.ood:
        print(f"Figuras del profesor desde {args.ood}")
        figP1_itersmin_vs_delta(args.ood, args.out)
        figP2_trazas(args.ood, args.out)
        figP3_certificado_vs_delta(args.ood, args.out)
        figP4_tiempo_vs_delta(args.ood, args.out)
        figP5_heatmap(args.ood, args.out)
    if not args.barrido and not args.ood:
        ap.error("pasa --barrido y/o --ood")


if __name__ == "__main__":
    main()
