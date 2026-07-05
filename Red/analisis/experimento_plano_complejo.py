"""
experimento_plano_complejo.py
=============================
Mapa de polos en el plano complejo: donde estan los autovalores de lazo abierto
y donde los coloca la red en lazo cerrado (K = Y Q^{-1}), sobre el politopo del
profesor. Es la vista geometrica de la LMI: estabilizar = meter TODOS los polos
al semiplano izquierdo (idealmente pasado la linea -alpha).

Todo en COORDENADAS NORMALIZADAS (cada politopo dividido por gamma=max|A,B|,
igual que en entrenamiento): asi la linea de margen -alpha es constante y es
consistente con la metrica de estabilizacion. La normalizacion es un escalado
uniforme (gamma>0), o sea NO cambia el signo de la parte real -> estable en
normalizado <=> estable en fisico.

Tres paneles:
  A. Migracion de lazo ABIERTO: los polos de cada vertice al crecer delta.
     El vertice 1 cruza al semiplano derecho (se desestabiliza); el 2 se va a
     la izquierda. Color = delta.
  B. Lazo CERRADO de la red (`control`) al maximo presupuesto: donde quedan los
     polos por delta. Se ve como se pegan a la frontera y saltan al RHP al fallar.
  C. Mismo sistema (delta=2), tres y_hat: control (falla, polo en RHP),
     control_margen (adentro), aleatorio (adentro). El y_hat decide.

Salidas en analisis/resultados/reporte/: plano_complejo.csv, fig_plano_complejo.*
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, torch, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

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
ALPHA = 0.01
ITERS = 8000


def normaliza(Ap, Bp):
    return bm.normalize_system(torch.as_tensor(Ap).double(), torch.as_tensor(Bp).double())


def K_de(model, An, Bn, y_hat=None, iters=ITERS):
    """Gain K = Y Q^{-1} tras `iters` de DR desde y_hat (None = el del encoder)."""
    Ab, Bb = An.unsqueeze(0), Bn.unsqueeze(0)
    model.use_implicit = False; model.eval()
    with torch.no_grad():
        if y_hat is None:
            y_hat = model(Ab, Bb, return_unconstrained=True)
        L, c, M_inv = model._dr_precompute(Ab, Bb)
        ck = dr_checkpoints(model, y_hat, L, c, M_inv, [iters], model.sigma)
        Q, Y = model._y_to_matrices(ck[iters][0])
    return Y[0] @ torch.linalg.inv(Q[0])


def eig_lazo_abierto(An):
    return [np.linalg.eigvals(An[i].numpy()) for i in range(An.shape[0])]


def eig_lazo_cerrado(An, Bn, K):
    return [np.linalg.eigvals((An[i] + Bn[i] @ K).numpy()) for i in range(An.shape[0])]


def eje_plano(ax, titulo, lims=(-4.5, 4.0, -4.0, 4.0)):
    ax.axvline(0, color="k", lw=1.1)                              # frontera de estabilidad
    ax.axvline(-ALPHA, ls="--", color="k", alpha=.5, lw=1)       # margen -alpha
    ax.axhline(0, color="gray", lw=.6, alpha=.5)
    ax.axvspan(0, lims[1], color="#a93226", alpha=.05)           # RHP = inestable
    ax.set(xlabel=r"Re$(\lambda)$", ylabel=r"Im$(\lambda)$", title=titulo,
           xlim=lims[:2], ylim=lims[2:])
    ax.set_aspect("equal", adjustable="box")


def obtiene_modelo(ls):
    """Entrena (y persiste) o carga los pesos. La persistencia hace la figura
    REPRODUCIBLE bit a bit: el fallo/exito de `control` en delta=2 es marginal
    (~±alpha) y varia entre corridas por el no-determinismo de BLAS multihilo."""
    from red.actuators import LMINetActuators
    pesos = OUT / f"modelo_{ls}_n2.pt"
    if pesos.exists():
        mdl = LMINetActuators(n=2, alpha=ALPHA, dr_iters=1000).double()
        mdl.load_state_dict(torch.load(pesos)); mdl.eval()
        print(f"  {ls}: pesos cargados de {pesos.name}")
        return mdl
    r = bm.run_experiment(arch="actuadores", n=2, N_list=[2, 3], m=1, dr_train=30,
                          dr_eval=1000, loss=ls, epochs=100, limit=150, seed=42,
                          compare_cvxpy=False, verbose=True)
    torch.save(r["model"].state_dict(), pesos)
    return r["model"]


def main():
    t0 = time.perf_counter()
    modelos = {ls: obtiene_modelo(ls) for ls in ["control", "control_margen"]}

    Ap0, Bp0 = bm.polytope_from_vertices([A1, A2], [[[0], [1]]])
    politopo = lambda d: bm.shift_poles(Ap0, d, directions=[D1, D2])
    deltas = np.linspace(0.0, 2.0, 21)
    filas = []

    fig = plt.figure(figsize=(16, 4.8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.4)
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2]); cax = fig.add_subplot(gs[3])
    cmap = plt.cm.viridis
    norm = plt.Normalize(0, 2)

    # ---------------- Panel A: lazo abierto ----------------
    eje_plano(axA, "A. Lazo abierto: migración de polos con $\\delta$")
    for d in deltas:
        An, Bn = normaliza(politopo(d), Bp0)
        for vi, ev in enumerate(eig_lazo_abierto(An)):
            mk = "o" if vi == 0 else "s"
            axA.scatter(ev.real, ev.imag, c=[cmap(norm(d))], marker=mk, s=26,
                        edgecolors="none", zorder=3)
            for e in ev:
                filas.append(dict(panel="A_abierto", serie=f"vertice{vi+1}", delta=d,
                                  re=float(e.real), im=float(e.imag)))
    axA.legend(handles=[Line2D([], [], marker="o", ls="", color="#440154", label="vértice 1 (se desestabiliza)"),
                        Line2D([], [], marker="s", ls="", color="#440154", label="vértice 2 (se estabiliza)")],
               fontsize=7.5, loc="upper left")

    # ---------------- Panel B: lazo cerrado red control ----------------
    eje_plano(axB, "B. Lazo cerrado de la red (`control`)")
    for d in deltas:
        An, Bn = normaliza(politopo(d), Bp0)
        K = K_de(modelos["control"], An, Bn)
        for vi, ev in enumerate(eig_lazo_cerrado(An, Bn, K)):
            mk = "o" if vi == 0 else "s"
            axB.scatter(ev.real, ev.imag, c=[cmap(norm(d))], marker=mk, s=26,
                        edgecolors="none", zorder=3)
            for e in ev:
                filas.append(dict(panel="B_cerrado_control", serie=f"vertice{vi+1}", delta=d,
                                  re=float(e.real), im=float(e.imag)))
    axB.annotate("los polos se AGOLPAN\nsobre el margen $-\\alpha$", xy=(-0.05, 2.7),
                 xytext=(-3.9, 3.0), fontsize=7.5, color="#333",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=.8))
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$\delta$ (desplazamiento)")

    # ---------------- Panel C: delta=2, tres y_hat (zoom en el eje) ----------------
    eje_plano(axC, "C. $\\delta=2$: dónde aterriza cada $\\hat{y}$", lims=(-0.55, 1.05, -2.2, 2.2))
    axC.set_aspect("auto")                                 # zoom fino en Re, no cuadrado
    An, Bn = normaliza(politopo(2.0), Bp0)
    rng = np.random.default_rng(0)
    dim = modelos["control"](An.unsqueeze(0), Bn.unsqueeze(0), return_unconstrained=True).shape[-1]
    casos = [("control", modelos["control"], None, "#a93226", "$\\hat{y}$ red (control)"),
             ("control_margen", modelos["control_margen"], None, "#1e8449", "$\\hat{y}$ red (control\\_margen)"),
             ("aleatorio", modelos["control"], torch.tensor(rng.standard_normal((1, dim)) * 0.3),
              "#7f8c8d", "$\\hat{y}$ aleatorio")]
    for tag, mdl, yh, col, lab in casos:
        K = K_de(mdl, An, Bn, y_hat=yh)
        evs = eig_lazo_cerrado(An, Bn, K)
        peor = max(e.real.max() for e in evs)
        for vi, ev in enumerate(evs):
            mk = "o" if vi == 0 else "s"
            axC.scatter(ev.real, ev.imag, color=col, marker=mk, s=70, zorder=3,
                        edgecolors="k", linewidths=.5,
                        label=f"{lab}  (peor Re={peor:+.3f})" if vi == 0 else None)
            for e in ev:
                filas.append(dict(panel="C_delta2", serie=f"{tag}_v{vi+1}", delta=2.0,
                                  re=float(e.real), im=float(e.imag)))
    axC.annotate("margen $-\\alpha$", xy=(-ALPHA, -2.05), fontsize=7.5, color="k",
                 rotation=90, va="bottom", ha="right")
    axC.legend(fontsize=7.2, loc="lower left")

    pd.DataFrame(filas).to_csv(OUT / "plano_complejo.csv", index=False)
    fig.savefig(OUT / "fig_plano_complejo.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig_plano_complejo.pdf", bbox_inches="tight")
    print(f"LISTO en {time.perf_counter() - t0:.0f}s -> fig_plano_complejo.png/.pdf")


if __name__ == "__main__":
    main()
