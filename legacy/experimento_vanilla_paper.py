"""
experimento_vanilla_paper.py
============================
Reproduccion de la receta del paper LMI-Net con la vanilla FIEL: LMINetVanilla con
sus defaults (MLP 64x64 ReLU sin LayerNorm, backward IMPLICITO, 500 iteraciones DR
fijas, epsilon=1e-3, sigma=0.01) entrenada con la perdida del paper (Ec. 21) via Adam.

Subconjunto: n_x=3, n_u=1, N=2. Evalua a {500,1000,2000,4000} iteraciones DR y reporta
% de sistemas estabilizados, % que cumple el decay alpha, y lambda_min(F) por sistema.

Regimen del paper: 1000 epocas (EPOCHS). Es LENTO (forward de 500 iters DR por lote);
baja EPOCHS/LIMIT para pruebas rapidas.

Salidas (analisis/resultados/experimentos/):
  - vanilla_paper.csv       resumen por presupuesto de iteraciones
  - vanilla_paper_sys.csv   por-sistema: estabilizado, decay, lambda_min(F)
  - fig_vanilla_paper.pdf
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt

from entrenamiento.training import load_vertices, split_items, make_batches, train
from red.vanilla import LMINetVanilla
from red.core import lmi_blocks

torch.set_default_dtype(torch.float64)

# ----------------------------- Config -------------------------------------
NX, NU, N = 3, 1, 2
LIMIT   = None          # 500 sistemas/celda (usa p.ej. 150 para pruebas rapidas)
EPOCHS  = 1000          # regimen del paper (parametrizable)
BATCH   = 16
SEED    = 42
DR_EVAL = [500, 1000, 2000, 4000]
OUT = Path(__file__).resolve().parents[1] / "analisis" / "resultados" / "experimentos"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})


@torch.no_grad()
def eval_budget(model, items, dr_iters):
    """Por sistema, a `dr_iters` iteraciones DR (forward desenrollado, sin grad):
    estabilizado, decay logrado y lambda_min(F(y*))."""
    old_it, old_impl = model.dr_iters, model.use_implicit
    model.dr_iters = dr_iters; model.use_implicit = False; model.eval()
    rows = []
    for A, B in make_batches(items, BATCH, shuffle=False):
        Q, Y = model(A, B)
        K = torch.bmm(Y, torch.linalg.inv(Q))
        F_list = lmi_blocks(Q, Y, A, B, model.alpha, model.epsilon)
        lam_min = torch.stack([torch.linalg.eigvalsh(0.5 * (F + F.transpose(-1, -2)))[..., 0]
                               for F in F_list], dim=-1).min(dim=-1).values          # (b,)
        for k in range(A.shape[0]):
            we = max(torch.linalg.eigvals(A[k, i] + B[k, i] @ K[k]).real.max().item()
                     for i in range(A.shape[1]))
            rows.append(dict(iters=dr_iters, worst_eig=we, decay=-we,
                             estabilizado=bool(we < 0), cumple_decay=bool(we <= -model.alpha),
                             lam_min_F=float(lam_min[k])))
    model.dr_iters, model.use_implicit = old_it, old_impl
    return rows


def main():
    items = load_vertices(NX, NU, N, limit=LIMIT)
    tr, te = split_items(items, seed=SEED)

    model = LMINetVanilla()          # defaults del paper, SIN overrides
    npar = sum(p.numel() for p in model.parameters())
    print(f"Vanilla FIEL al paper: {npar} parametros | backprop={model.backprop} | "
          f"dr_iters={model.dr_iters} | epsilon={model.epsilon} | alpha={model.alpha}")
    print(f"Entrenando {EPOCHS} epocas sobre {len(tr)} sistemas (n_x={NX}, n_u={NU}, N={N})"
          f" con perdida del paper (Ec. 21)...")
    t0 = time.perf_counter()
    train(model, {(NU, N): tr}, epochs=EPOCHS, batch=BATCH, seed=SEED,
          log_every=max(1, EPOCHS // 10))
    print(f"  train {time.perf_counter() - t0:.0f}s")

    sys_rows = []
    for dr in DR_EVAL:
        sys_rows += eval_budget(model, te, dr)
    df = pd.DataFrame(sys_rows); df.to_csv(OUT / "vanilla_paper_sys.csv", index=False)

    summ = df.groupby("iters").agg(
        estable_pct=("estabilizado", lambda s: 100 * s.mean()),
        decay_pct=("cumple_decay", lambda s: 100 * s.mean()),
        lam_min_F_medio=("lam_min_F", "mean"),
        lam_min_F_peor=("lam_min_F", "min"),
        n=("estabilizado", "size")).reset_index()
    summ.to_csv(OUT / "vanilla_paper.csv", index=False)
    print("\n=========  VANILLA FIEL (paper) — evaluacion por iteraciones DR  =========")
    print(summ.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(summ.iters, summ.estable_pct, marker="o", label="% estabilizados")
    ax.plot(summ.iters, summ.decay_pct, marker="s", label=f"% cumple decay α={model.alpha}")
    ax.set(xlabel="iteraciones DR (evaluacion)", ylabel="%", ylim=(0, 100),
           title="Vanilla fiel al paper ($n_x=3,\\ n_u=1,\\ N=2$)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_vanilla_paper.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
