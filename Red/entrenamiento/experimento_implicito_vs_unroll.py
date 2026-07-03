"""
experimento_implicito_vs_unroll.py
==================================
Compara entrenar el modelo de actuadores con:
  - UNROLLING  : backward por las 30 iteraciones desenrolladas (DR_TR=30)
  - IMPLICITA  : backward por diferenciacion implicita (gradiente del punto fijo)

Mismo init/seed, mismos datos (n_x=3, conjunto m in {1,2}, N in {2,3,4,5}).
Para aislar el efecto del gradiente, AMBOS modelos se evaluan con el MISMO
forward estandar (DR desenrollado a 1000 iters, sin gradiente).

Salidas en analisis/actuadores/:
  - implicito_vs_unroll.csv    estabilizacion por (metodo, m, N) + tiempo de train
  - fig_impl_vs_unroll.pdf
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt

from entrenamiento.training import load_vertices, split_items, train, control_loss, evaluate
from red.actuators import LMINetActuators

torch.set_default_dtype(torch.float64)

NX, NLIST, MLIST = 3, [2, 3, 4, 5], [1, 2]
LIMIT, EPOCHS, DR_TR, DR_EV, BATCH, SEED = 150, 15, 30, 1000, 16, 42
IMPL_MAXIT, IMPL_TOL = 2000, 1e-8
OUT = Path(__file__).resolve().parents[1] / "analisis" / "resultados" / "experimentos"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})


def build():
    torch.manual_seed(SEED)
    return LMINetActuators(n=NX, alpha=0.01, dr_iters=DR_TR).double()


def eval_all(model, data):
    model.set_implicit(False)                    # eval justo: forward estandar para ambos
    rows = []
    for (m, N), (_, te) in data.items():
        r = evaluate(model, te, order=NX, alpha=0.01, dr_eval=DR_EV, batch=BATCH)
        rows.append(dict(m=m, N=N, estable_pct=r["stable_pct"]))
    return pd.DataFrame(rows)


def main():
    data = {(m, N): split_items(load_vertices(NX, m, N, limit=LIMIT), seed=SEED)
            for m in MLIST for N in NLIST}
    train_by = {k: data[k][0] for k in data}

    print("=== UNROLLING (DR_TR=30) ===")
    mU = build()
    t0 = time.perf_counter(); train(mU, train_by, epochs=EPOCHS, batch=BATCH, seed=SEED, loss_fn=control_loss)
    tU = time.perf_counter() - t0

    print("\n=== IMPLICITA (punto fijo) ===")
    mI = build().set_implicit(True)
    mI.implicit_max_iters = IMPL_MAXIT; mI.implicit_tol = IMPL_TOL
    t0 = time.perf_counter(); train(mI, train_by, epochs=EPOCHS, batch=BATCH, seed=SEED, loss_fn=control_loss)
    tI = time.perf_counter() - t0

    dfU = eval_all(mU, data); dfU["metodo"] = "unrolling"
    dfI = eval_all(mI, data); dfI["metodo"] = "implicita"
    df = pd.concat([dfU, dfI], ignore_index=True)
    df.to_csv(OUT / "implicito_vs_unroll.csv", index=False)

    print("\n================  ESTABILIZACION (%)  ================")
    for tag, sub in df.groupby("metodo"):
        piv = sub.pivot(index="m", columns="N", values="estable_pct")
        print(f"\n[{tag}]  (train {tU if tag=='unrolling' else tI:.0f}s)")
        print(piv.round(1).to_string())
        print("  media por m:", {int(m): round(sub[sub.m == m].estable_pct.mean(), 1) for m in MLIST})

    print(f"\nMEDIA GLOBAL:  unrolling={dfU.estable_pct.mean():.1f}%   "
          f"implicita={dfI.estable_pct.mean():.1f}%   (dif={dfI.estable_pct.mean()-dfU.estable_pct.mean():+.1f} pts)")
    print(f"TIEMPO TRAIN:  unrolling={tU:.0f}s   implicita={tI:.0f}s   ({tI/tU:.1f}x)")

    # figura
    fig, ax = plt.subplots(figsize=(6.2, 3.6)); x = np.arange(len(MLIST)); w = .36
    for i, tag in enumerate(["unrolling", "implicita"]):
        vals = [df[(df.metodo == tag) & (df.m == m)].estable_pct.mean() for m in MLIST]
        b = ax.bar(x + (i - .5) * w, vals, w, label=tag, color=["#7f8c8d", "#1e8449"][i], alpha=.85)
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"m={m}" for m in MLIST]); ax.set_ylim(0, 100)
    ax.set_ylabel("% estabilizados"); ax.legend()
    ax.set_title("Gradiente implícito vs unrolling-30 ($n_x=3$, media sobre N)")
    fig.tight_layout(); fig.savefig(OUT / "fig_impl_vs_unroll.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
