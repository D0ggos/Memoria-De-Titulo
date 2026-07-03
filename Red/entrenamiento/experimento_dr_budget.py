"""
experimento_dr_budget.py
========================
¿Cuanto del 16.5% de fallos del modelo CONJUNTO se recupera con mas iteraciones
de Douglas-Rachford, y cuanto es "duro de verdad"?

Entrena una vez el modelo conjunto por orden n_x in {3,4,5} y evalua la
estabilizacion en varios presupuestos DR {500,1000,2000,4000} usando UNA sola
pasada con checkpoints (barato). Clasifica cada sistema de test en:
  - siempre-ok   : estable ya a 500 iters
  - recuperado   : falla a 500 pero estable a 4000
  - duro         : falla aun a 4000
y compara su condicionamiento kappa(A) (sospechoso principal).

Salidas en analisis/actuadores/:
  - dr_budget.csv        estabilizacion (%) por presupuesto (agregado y por orden)
  - dr_budget_sys.csv    por-sistema: estable en cada presupuesto + condA, normB
  - fig_dr_budget.pdf    curva fallo vs iteraciones + kappa por categoria
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt

from entrenamiento.training import load_vertices, split_items, train, control_loss, make_batches
from red.actuators import LMINetActuators
from analisis.validate_projection import build_operator
from analisis.benchmark_dr_vs_cvxpy import dr_checkpoints, per_system_metrics

torch.set_default_dtype(torch.float64)

ORDERS  = [3, 4, 5]
NLIST   = [2, 3, 4, 5]
MLIST   = [1, 2]
BUDGETS = [500, 1000, 2000, 4000, 10000]
LIMIT   = None
EPOCHS  = 15
DR_TR   = 30
BATCH   = 16
SEED    = 42
OUT = Path(__file__).resolve().parents[1] / "analisis" / "resultados" / "experimentos"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})


def build(nx):
    torch.manual_seed(SEED)
    return LMINetActuators(n=nx, alpha=0.01, dr_iters=DR_TR).double()


@torch.no_grad()
def eval_budgets(model, items, nx, m, N):
    rows = []
    for A, B in make_batches(items, BATCH, shuffle=False):
        y_hat = model(A, B, return_unconstrained=True)          # warm-start del encoder
        L, c, M_inv = build_operator(model, A, B)
        ck = dr_checkpoints(model, y_hat, L, c, M_inv, BUDGETS, model.sigma)
        stab = {}
        for bud in BUDGETS:
            _, cl = per_system_metrics(model, ck[bud][0], A, B)
            stab[bud] = cl < 0
        for k in range(A.shape[0]):
            condA = max(np.linalg.cond(A[k, i].numpy()) for i in range(N))
            normB = max(np.linalg.norm(B[k, i].numpy(), 2) for i in range(N))
            rows.append(dict(order=nx, m=m, N=N, condA=condA, normB=normB,
                             **{f"stab_{b}": bool(stab[b][k]) for b in BUDGETS}))
    return rows


def main():
    sys_rows = []
    for nx in ORDERS:
        print(f"\n########## ORDEN n_x = {nx} ##########")
        data = {(m, N): split_items(load_vertices(nx, m, N, limit=LIMIT), seed=SEED)
                for m in MLIST for N in NLIST}
        mJ = build(nx)
        train(mJ, {k: data[k][0] for k in data}, epochs=EPOCHS, batch=BATCH, seed=SEED, loss_fn=control_loss)
        for (m, N), (_, te) in data.items():
            sys_rows += eval_budgets(mJ, te, nx, m, N)
        pd.DataFrame(sys_rows).to_csv(OUT / "dr_budget_sys.csv", index=False)
        print(f"n_x={nx} listo.")

    df = pd.DataFrame(sys_rows)
    # tasa de fallo por presupuesto (agregada)
    agg = pd.DataFrame({"iters": BUDGETS,
                        "fallo_pct": [100 * (1 - df[f"stab_{b}"].mean()) for b in BUDGETS]})
    agg.to_csv(OUT / "dr_budget.csv", index=False)

    # categorias
    df["cat"] = np.where(df[f"stab_{BUDGETS[0]}"], "siempre-ok",
                np.where(df[f"stab_{BUDGETS[-1]}"], "recuperado", "duro"))
    _report(df, agg)
    _plots(df, agg)


def _report(df, agg):
    print("\n================  FALLO (%) vs ITERACIONES DR  ================")
    print(agg.to_string(index=False))
    print("\n=== categorias (sobre", len(df), "sistemas de test) ===")
    print((df.cat.value_counts(normalize=True) * 100).round(1).to_string())
    print("\n=== kappa(A) mediana por categoria ===")
    print(df.groupby("cat").condA.median().round(1).to_string())
    print("\n=== normB mediana por categoria ===")
    print(df.groupby("cat").normB.median().round(3).to_string())
    hard = df[df.cat == "duro"]
    print(f"\nNucleo DURO: {len(hard)} sistemas ({len(hard)/len(df)*100:.1f}%) siguen sin "
          f"estabilizar a {BUDGETS[-1]} iters.  kappa mediana={hard.condA.median():.1f} "
          f"(vs {df[df.cat=='siempre-ok'].condA.median():.1f} de los siempre-ok)")


def _plots(df, agg):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    axs[0].plot(agg.iters, agg.fallo_pct, marker="o", color="#c0392b")
    axs[0].set(xlabel="iteraciones DR", ylabel="% sistemas NO estabilizados",
               title="El fallo baja con mas iteraciones\n(pero se estanca en un nucleo duro)")
    for _, r in agg.iterrows():
        axs[0].annotate(f"{r.fallo_pct:.1f}%", (r.iters, r.fallo_pct), fontsize=8,
                        textcoords="offset points", xytext=(0, 6))
    cats = ["siempre-ok", "recuperado", "duro"]
    data = [np.log10(df[df.cat == c].condA.clip(1e-3)) for c in cats]
    bp = axs[1].boxplot(data, tick_labels=cats, showfliers=False, patch_artist=True)
    for box, col in zip(bp["boxes"], ["#2471a3", "#d68910", "#c0392b"]):
        box.set(facecolor=col, alpha=.6)
    axs[1].set(ylabel=r"$\log_{10}\,\kappa(A)$", title="Los sistemas duros son los peor condicionados")
    fig.tight_layout(); fig.savefig(OUT / "fig_dr_budget.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
