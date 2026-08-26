"""Figura COMPUTADA: conjunto factible real + trayectoria real de Douglas-Rachford.

Un solo panel, instancia nominal. C vive en R^5 (n_x=2: Q simetrica -> 3, Y 1x2 -> 2), asi
que se corta por el plano de 2 componentes principales de la nube {y_hat, trayectoria, y*,
y_cvx}; se anota la varianza explicada. La region dibujada es {y en el plano :
lam_min(F(y)) >= 0} evaluada exactamente, no un esquema.
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                  # .../Red
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow

torch.set_default_dtype(torch.float64); torch.set_num_threads(4)

from red.core import lmi_blocks
from barrido.ood_eval import load_checkpoint
from barrido.ood_banks import professor_system
from analisis.benchmark import _cvxpy_solve
from entrenamiento.training import get_loss_fn

OUT = ROOT / "analisis" / "resultados" / "barrido" / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
# Checkpoint de un modelo n_x=2 (los del barrido viven en <etapa>/models/).
CKPT = os.environ.get("LMINET_CKPT",
    "analisis/resultados/barrido/E2/models/"
    "E2__actuadores__nx2__margen_norm__unrolling__dr30__a0p01__sz150__s42/epoch_400.pt")
DELTA = 0.0
N_DR, G = 400, 460

# gradientes de perdida a dibujar: nombre -> (color, etiqueta)
GRADS = {"margen_norm":      ("tab:green",  r"$-\nabla\,\mathcal{L}_{\mathrm{margen\_norm}}$"),
         "condicionamiento": ("tab:purple", r"$-\nabla\,\mathcal{L}_{\mathrm{condicionamiento}}$"),
         "control":          ("tab:red",    r"$-\nabla\,\mathcal{L}_{\mathrm{control}}$"),
         "esfuerzo":         ("tab:orange", r"$-\nabla\,\mathcal{L}_{\mathrm{esfuerzo}}$")}


def y_from_QY(model, Q, Y):
    ti = model.triu_indices
    return torch.cat([Q[:, ti[0], ti[1]], Y.reshape(Q.shape[0], -1)], dim=-1)


def campo(model, Ygrid, A, B):
    """lam_min(F) y kappa(Q) sobre un lote de y."""
    Gn = Ygrid.shape[0]
    Q, Yv = model._y_to_matrices(Ygrid)
    Ab = A.unsqueeze(0).expand(Gn, -1, -1, -1); Bb = B.unsqueeze(0).expand(Gn, -1, -1, -1)
    Fs = lmi_blocks(Q, Yv, Ab, Bb, model.alpha, model.epsilon)
    lam = torch.stack([torch.linalg.eigvalsh(0.5 * (F + F.transpose(-1, -2)))[..., 0]
                       for F in Fs], dim=-1).min(dim=-1).values
    ev = torch.linalg.eigvalsh(Q)
    return lam, ev[..., -1] / ev[..., 0].clamp_min(1e-12)


def main():
    model, cfg, ep = load_checkpoint(CKPT)
    A, B = professor_system(DELTA)
    Ab, Bb = A.unsqueeze(0), B.unsqueeze(0)

    with torch.no_grad():
        y_hat = model(Ab, Bb, return_unconstrained=True)
        L, c, M_inv = model._dr_precompute(Ab, Bb)
        y_k = y_hat.clone(); x_k = torch.bmm(L, y_k.unsqueeze(-1)).squeeze(-1) + c
        traj = [model._dr_final_proj(y_hat, y_k, x_k, L, c, M_inv).squeeze(0).clone()]
        for _ in range(N_DR):
            y_k, x_k = model._dr_step_batch(y_k, x_k, y_hat, L, c, M_inv)
            traj.append(model._dr_final_proj(y_hat, y_k, x_k, L, c, M_inv).squeeze(0).clone())
        traj = torch.stack(traj)
    y_star = traj[-1]
    feas, _, Qc, Yc = _cvxpy_solve(model, A, B)
    y_cvx = y_from_QY(model, Qc, Yc).squeeze(0)

    # plano PCA
    M = torch.cat([traj, y_hat, y_star.unsqueeze(0), y_cvx.unsqueeze(0)], dim=0)
    o = y_star; Mc = M - o
    _, S, Vh = torch.linalg.svd(Mc, full_matrices=False)
    e1, e2 = Vh[0], Vh[1]
    var = float((S[:2] ** 2).sum() / (S ** 2).sum())

    def co(y):
        d = y - o
        u, v = float(d @ e1), float(d @ e2)
        return u, v, float((d - u * e1 - v * e2).norm())

    p_hat, p_star, p_cvx = co(y_hat.squeeze(0)), co(y_star), co(y_cvx)
    dist_cvx = float((y_cvx - y_star).norm())

    R = float(max(abs(Mc @ e1).max(), abs(Mc @ e2).max())) * 1.45
    us = torch.linspace(-R, R, G)
    U, V = torch.meshgrid(us, us, indexing="xy")
    Yg = o.unsqueeze(0) + U.reshape(-1, 1) * e1.unsqueeze(0) + V.reshape(-1, 1) * e2.unsqueeze(0)
    with torch.no_grad():
        lam, kap = campo(model, Yg, A, B)
    lam = lam.reshape(G, G).numpy(); kap = kap.reshape(G, G).numpy()
    kin = np.where(lam >= 0, kap, np.nan)

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    lk = np.log10(np.clip(kin, 1, None))
    # rasterized: el mapa de color son ~2e5 celdas; en vectorial pesaria 4 MB.
    im = ax.pcolormesh(U.numpy(), V.numpy(), lk, cmap="viridis_r", shading="auto",
                       vmin=0, vmax=np.nanpercentile(lk, 99), alpha=.92, rasterized=True)
    ax.contour(U.numpy(), V.numpy(), lam, levels=[0.0], colors="k", linewidths=1.8)

    tj = np.array([co(y)[:2] for y in traj])
    ax.plot(tj[:, 0], tj[:, 1], "-", color="crimson", lw=1.3, zorder=4)
    for i in (2, 6, 14, 30):                       # flechitas de sentido en la trayectoria
        ax.annotate("", xy=tj[i + 1, :2], xytext=tj[i, :2], zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="crimson", lw=1.3))

    esc = R * .30
    for nombre, (col, _) in GRADS.items():
        y = y_star.clone().unsqueeze(0).requires_grad_(True)
        Qq, Yy = model._y_to_matrices(y)
        val = get_loss_fn(nombre, model)(Qq, Yy, Ab, Bb)
        g, = torch.autograd.grad(val, y)
        gu, gv = float(g.squeeze(0) @ e1), float(g.squeeze(0) @ e2)
        nr = (gu ** 2 + gv ** 2) ** .5
        if nr < 1e-12:
            continue
        ax.arrow(p_star[0], p_star[1], -gu / nr * esc, -gv / nr * esc, width=R * .005,
                 head_width=R * .028, color=col, alpha=.9, zorder=7, length_includes_head=True)

    ax.plot(p_hat[0], p_hat[1], "o", color="black", ms=8, mec="white", mew=1, zorder=8)
    ax.plot(p_cvx[0], p_cvx[1], "s", mfc="none", mec="tab:blue", mew=2.2, ms=17, zorder=8)
    ax.plot(p_star[0], p_star[1], "*", color="crimson", ms=17, mec="white", mew=.9, zorder=9)

    leg = [Line2D([], [], marker="o", color="k", ls="", ms=8, label=r"$\hat{y}$: propuesta del codificador"),
           Line2D([], [], color="crimson", lw=1.3, label=f"trayectoria de Douglas--Rachford ({N_DR} iter.)"),
           Line2D([], [], marker="*", color="crimson", ls="", ms=15, label=r"$y^*$: salida del solucionador"),
           Line2D([], [], marker="s", mfc="none", mec="tab:blue", mew=2, ls="", ms=12,
                  label=r"$y_{\mathrm{cvx}}$: certificado convexo"),
           Line2D([], [], color="k", lw=1.8, label=r"frontera $\lambda_{\min}(F)=0$")]
    leg += [Line2D([], [], color=c, lw=3, label=t) for c, t in GRADS.values()]
    ax.legend(handles=leg, fontsize=8, loc="upper left", framealpha=.93)

    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
    cb.set_label(r"$\log_{10}\kappa(Q)$ dentro de $\mathcal{C}$", fontsize=9)
    ax.set_xlabel("primera componente principal del plano de corte", fontsize=9)
    ax.set_ylabel("segunda componente principal", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_title("Conjunto factible y trayectoria del solucionador, calculados sobre una instancia real",
                 fontsize=10.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_geometria_computada.{ext}", dpi=180, bbox_inches="tight")

    print(f"varianza explicada por el plano : {100*var:.2f} %")
    print(f"||y_cvx - y*||                  : {dist_cvx:.3e}   <- por eso se superponen")
    print(f"componente de y_hat fuera del plano: {p_hat[2]:.2e}")
    print(f"area factible del corte         : {100*np.mean(lam>=0):.2f} %")
    print(f"kappa(Q) dentro de C            : mediana {np.nanmedian(kin):.3g}")
    print(f"escritas: {OUT/'fig_geometria_computada.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
