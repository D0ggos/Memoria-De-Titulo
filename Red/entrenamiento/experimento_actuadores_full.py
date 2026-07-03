"""
experimento_actuadores_full.py
==============================
Version COMPLETA (toda la base) del experimento de invarianza de actuadores.

Para cada orden n_x in {3,4,5} (los que tienen m=1 Y m=2) entrena la
arquitectura invariante a N y equivariante a n_u en dos regimenes:
  A) CONJUNTO  : m in {1,2}, N in {2,3,4,5}, TODOS los 500 sistemas/celda
  B) ZERO-SHOT : entrena SOLO m=1, evalua en m=2 (no visto)

Guarda por-sistema los casos NO estabilizados por el modelo CONJUNTO (el
desplegable) con sus descriptores, y analiza posibles causas (fallidos vs ok).

Salidas en analisis/actuadores/:
  - result.csv       estabilizacion por (order, modelo, m, N)
  - all_desc.csv     descriptores por-sistema (conjunto) + estabilizado
  - failures.csv     solo los NO estabilizados + descriptores
  - failures.npz     matrices A,B crudas de esos sistemas
  - fig_estab.pdf    estabilizacion por orden y m (conjunto vs zero-shot)
  - fig_causas.pdf   descriptores: fallidos vs estabilizados
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("pdf")
import matplotlib.pyplot as plt

from entrenamiento.training import load_vertices, split_items, train, control_loss, make_batches
from red.actuators import LMINetActuators

torch.set_default_dtype(torch.float64)

ORDERS = [3, 4, 5]
NLIST  = [2, 3, 4, 5]
MLIST  = [1, 2]
LIMIT  = None        # None = los 500 sistemas por celda
EPOCHS = 15
DR_TR  = 30
DR_EV  = 1000
BATCH  = 16
SEED   = 42

OUT = Path(__file__).resolve().parents[1] / "analisis" / "resultados" / "experimentos"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})


def build(nx):
    torch.manual_seed(SEED)
    return LMINetActuators(n=nx, alpha=0.01, dr_iters=DR_TR).double()


def descriptors_row(A, B, N):
    Al = [A[i].numpy() for i in range(N)]; Bl = [B[i].numpy() for i in range(N)]
    Abar = np.mean(Al, 0)
    return dict(absc_ol=max(np.linalg.eigvals(Ai).real.max() for Ai in Al),
                normA=max(np.linalg.norm(Ai, 2) for Ai in Al),
                normB=max(np.linalg.norm(Bi, 2) for Bi in Bl),
                condA=max(np.linalg.cond(Ai) for Ai in Al),
                dispA=max(np.linalg.norm(Ai - Abar, 2) for Ai in Al))


@torch.no_grad()
def per_system_eval(model, items, nx, m, N, collect_desc=False):
    """Lista de dicts por sistema: estabilizado + worst + (opc.) descriptores y A,B."""
    old = model.dr_iters; model.dr_iters = DR_EV; model.eval()
    out = []
    for A, B in make_batches(items, BATCH, shuffle=False):
        Q, Y = model(A, B)
        K = torch.bmm(Y, torch.linalg.inv(Q))                 # (b, m, n)
        for k in range(A.shape[0]):
            we = max(torch.linalg.eigvals(A[k, i] + B[k, i] @ K[k]).real.max().item()
                     for i in range(N))
            row = dict(order=nx, m=m, N=N, worst=we, stabilized=bool(we < 0))
            if collect_desc:
                row.update(descriptors_row(A[k], B[k], N))
                row["_A"] = A[k].numpy(); row["_B"] = B[k].numpy()
            out.append(row)
    model.dr_iters = old
    return out


def main():
    res_rows, conj_rows, fail_AB = [], [], []

    for nx in ORDERS:
        print(f"\n########## ORDEN n_x = {nx} ##########")
        data = {(m, N): split_items(load_vertices(nx, m, N, limit=LIMIT), seed=SEED)
                for m in MLIST for N in NLIST}
        keys_all = list(data.keys())
        keys_m1  = [k for k in keys_all if k[0] == 1]

        print("--- CONJUNTO (m in {1,2}) ---")
        mJ = build(nx)
        train(mJ, {k: data[k][0] for k in keys_all}, epochs=EPOCHS, batch=BATCH, seed=SEED, loss_fn=control_loss)

        print("--- ZERO-SHOT (solo m=1) ---")
        mZ = build(nx)
        train(mZ, {k: data[k][0] for k in keys_m1}, epochs=EPOCHS, batch=BATCH, seed=SEED, loss_fn=control_loss)

        for (m, N), (_, te) in data.items():
            evJ = per_system_eval(mJ, te, nx, m, N, collect_desc=True)
            evZ = per_system_eval(mZ, te, nx, m, N, collect_desc=False)
            for tag, ev in [("conjunto", evJ), ("zero-shot (train m=1)", evZ)]:
                res_rows.append(dict(order=nx, modelo=tag, m=m, N=N,
                                     estable_pct=np.mean([r["stabilized"] for r in ev]) * 100,
                                     n=len(ev)))
            for r in evJ:
                A_, B_ = r.pop("_A"), r.pop("_B")
                conj_rows.append(r)                       # descriptores + stabilized
                if not r["stabilized"]:
                    fail_AB.append((nx, m, N, A_, B_))

        pd.DataFrame(res_rows).to_csv(OUT / "result.csv", index=False)
        print(f"n_x={nx} listo. Fallos (conjunto) acumulados: {len(fail_AB)}")

    res = pd.DataFrame(res_rows); res.to_csv(OUT / "result.csv", index=False)
    alld = pd.DataFrame(conj_rows); alld.to_csv(OUT / "all_desc.csv", index=False)
    fails = alld[~alld.stabilized].copy(); fails.to_csv(OUT / "failures.csv", index=False)
    np.savez_compressed(OUT / "failures.npz",
                        meta=np.array([(nx, m, N) for nx, m, N, _, _ in fail_AB]),
                        **{f"A_{i}": AB[3] for i, AB in enumerate(fail_AB)},
                        **{f"B_{i}": AB[4] for i, AB in enumerate(fail_AB)})

    _report(res, alld)
    _plots(res, alld)


def _report(res, alld):
    print("\n================  ESTABILIZACION (%)  ================")
    for tag in res.modelo.unique():
        print(f"\n[{tag}]  (media sobre N)")
        print(res[res.modelo == tag].groupby(["order", "m"]).estable_pct.mean()
              .unstack("m").round(1).to_string())
    zc = res[res.modelo.str.startswith("zero") & (res.m == 2)].estable_pct.mean()
    jc = res[(res.modelo == "conjunto") & (res.m == 2)].estable_pct.mean()
    print(f"\nGENERALIZACION cross-actuador (m=2, todos los ordenes): "
          f"zero-shot={zc:.1f}%  vs entrenado={jc:.1f}%  (gap={jc - zc:+.1f} pts)")

    ok, bad = alld[alld.stabilized], alld[~alld.stabilized]
    print(f"\nFallos (conjunto): {len(bad)} de {len(alld)} test ({len(bad)/len(alld)*100:.1f}%)")
    print("\n=== POSIBLES CAUSAS: mediana fallido vs estabilizado (datos normalizados) ===")
    for c in ["condA", "normB", "dispA", "absc_ol", "normA"]:
        rb, ro = bad[c].median(), ok[c].median()
        print(f"  {c:8}: fallido={rb:.3g}  ok={ro:.3g}  ratio={rb/ro:.2f}x")
    print("\n=== tasa de fallo por (m, N) ===")
    print((100 * (1 - alld.groupby(["m", "N"]).stabilized.mean())).round(1).unstack("N").to_string())


def _plots(res, alld):
    fig, axs = plt.subplots(1, len(ORDERS), figsize=(9, 3.2), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, nx in zip(axs, ORDERS):
        sub = res[res.order == nx]; x = np.arange(len(MLIST)); w = .36
        for i, tag in enumerate(["conjunto", "zero-shot (train m=1)"]):
            vals = [sub[(sub.modelo == tag) & (sub.m == m)].estable_pct.mean() for m in MLIST]
            b = ax.bar(x + (i - .5) * w, vals, w, label=tag,
                       color=["#2471a3", "#c0392b"][i], alpha=.85)
            ax.bar_label(b, fmt="%.0f", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([f"m={m}" for m in MLIST])
        ax.set_title(f"$n_x={nx}$"); ax.set_ylim(0, 100)
    axs[0].set_ylabel("% estabilizados"); axs[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Generalización cross-actuador por orden (media sobre N)", y=1.02)
    fig.tight_layout(); fig.savefig(OUT / "fig_estab.pdf"); plt.close(fig)

    ok, bad = alld[alld.stabilized], alld[~alld.stabilized]
    if len(bad):
        fig, axs = plt.subplots(1, 4, figsize=(11, 3))
        for ax, col, t, lg in [(axs[0], "condA", r"$\kappa(A)$", True),
                               (axs[1], "normB", r"$\|B\|$", False),
                               (axs[2], "dispA", "dispersión politopo", False),
                               (axs[3], "absc_ol", "abscisa lazo abierto", False)]:
            fo = np.log10(ok[col].clip(1e-3)) if lg else ok[col]
            fb = np.log10(bad[col].clip(1e-3)) if lg else bad[col]
            ax.boxplot([fo, fb], tick_labels=["estab.", "fallido"], showfliers=False, patch_artist=True)
            ax.set_title(("$\\log_{10}$ " if lg else "") + t)
        fig.suptitle("Posibles causas: ¿qué distingue a los sistemas no estabilizados?", y=1.02)
        fig.tight_layout(); fig.savefig(OUT / "fig_causas.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
