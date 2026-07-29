"""
experimento_unroll_iters.py
===========================
Barre las iteraciones de DR usadas en ENTRENAMIENTO (unrolling) y mide la
estabilizacion. Prueba la hipotesis: mas iteraciones de unrolling se acercan al
comportamiento de la proyeccion exacta (implicita) y empeoran por explotacion
del borde factible.

Mismo harness que experimento_implicito_vs_unroll.py (n_x=3, conjunto, LIMIT=150,
EPOCHS=15, seed=42, eval a DR_EV=1000). Referencia: implicita = 60.4%.
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
LIMIT, EPOCHS, DR_EV, BATCH, SEED = 150, 15, 1000, 16, 42
DR_TR_LIST = [30, 500]
OUT = Path(__file__).resolve().parents[1] / "analisis" / "resultados" / "experimentos"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})


def main():
    data = {(m, N): split_items(load_vertices(NX, m, N, limit=LIMIT), seed=SEED)
            for m in MLIST for N in NLIST}
    train_by = {k: data[k][0] for k in data}

    rows = []
    for dr_tr in DR_TR_LIST:
        print(f"\n=== UNROLLING DR_TR={dr_tr} ===")
        torch.manual_seed(SEED)
        model = LMINetActuators(n=NX, alpha=0.01, dr_iters=dr_tr).double()
        t0 = time.perf_counter(); train(model, train_by, epochs=EPOCHS, batch=BATCH, seed=SEED, loss_fn=control_loss)
        tt = time.perf_counter() - t0
        for (m, N), (_, te) in data.items():
            r = evaluate(model, te, order=NX, alpha=0.01, dr_eval=DR_EV, batch=BATCH)
            rows.append(dict(dr_tr=dr_tr, m=m, N=N, estable_pct=r["stable_pct"], train_s=tt))
        sub = pd.DataFrame([x for x in rows if x["dr_tr"] == dr_tr])
        print(f"  media global={sub.estable_pct.mean():.1f}%  (train {tt:.0f}s)")

    df = pd.DataFrame(rows); df.to_csv(OUT / "unroll_iters.csv", index=False)
    print("\n================  ESTABILIZACION (%)  ================")
    for dr_tr in DR_TR_LIST:
        sub = df[df.dr_tr == dr_tr]
        print(f"\n[DR_TR={dr_tr}]  media={sub.estable_pct.mean():.1f}%  "
              f"(m=1:{sub[sub.m==1].estable_pct.mean():.1f}  m=2:{sub[sub.m==2].estable_pct.mean():.1f})")
        print(sub.pivot(index="m", columns="N", values="estable_pct").round(1).to_string())

    # figura: media global vs DR_TR de entrenamiento, con implicita de referencia
    fig, ax = plt.subplots(figsize=(6, 3.6))
    g = df.groupby("dr_tr").estable_pct.mean()
    ax.plot(g.index, g.values, marker="o", color="#7f8c8d", label="unrolling")
    for x, y in g.items():
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=9)
    ax.axhline(60.4, ls="--", color="#1e8449", label="implícita (≈∞ iters) = 60.4%")
    ax.set(xlabel="iteraciones DR en ENTRENAMIENTO", ylabel="% estabilizados (eval a 1000)",
           title="Mas iteraciones de unrolling → peor (converge a la implícita)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_unroll_iters.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
