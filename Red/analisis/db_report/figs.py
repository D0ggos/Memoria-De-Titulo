import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

S = "."
OUT = "figuras"
import os; os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = dict(ol="#c0392b", cl="#2471a3", acc="#1e8449", g="#7d3c98", o="#d68910")
df = pd.read_csv(f"{S}/descriptors.csv")
A = np.load(f"{S}/arrays.npz")

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", dpi=200); plt.close(fig)
    print("ok", name)

# 1 — Cobertura factorial (conteos por nu)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, nu in zip(axs, [1, 2]):
    sub = df[df.nu == nu]
    M = sub.groupby(['nx', 'N']).size().unstack('N')
    im = ax.imshow(M.values, cmap="Blues", vmin=0, vmax=500, aspect="auto")
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index)
    ax.set_xlabel("$N$ (vértices)"); ax.set_ylabel("$n_x$ (orden)")
    ax.set_title(f"$n_u={nu}$  ({len(sub)} sistemas)")
    for (i, j), v in np.ndenumerate(M.values):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=8)
    ax.grid(False)
fig.suptitle("Cobertura: nº de sistemas por $(n_x, N)$", y=1.02)
save(fig, "fig1_cobertura")

# 2 — Estabilidad lazo abierto vs cerrado (abscisa espectral, peor vértice)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.hist(df.absc_ol_max, bins=np.linspace(-20, 200, 80), color=C["ol"], alpha=.8)
ax.axvline(0, color="k", lw=1, ls="--")
ax.text(0.26, 0.50, f"{(df.absc_ol_max>0).mean()*100:.1f}% inestable\nen lazo abierto",
        transform=ax.transAxes, fontsize=9, color=C["ol"])
ax.set_xlabel(r"abscisa espectral del peor vértice  $\max_i\,\max\,\mathrm{Re}\,\lambda(A_i)$")
ax.set_ylabel("nº de sistemas"); ax.set_title("Planta inestable (rojo) vs. lazo cerrado (azul, inset)")
axin = ax.inset_axes([0.50, 0.40, 0.46, 0.48])
axin.hist(df.absc_cl_max, bins=np.linspace(-2, 0.05, 60), color=C["cl"])
axin.axvline(0, color="k", lw=1, ls="--")
axin.set_title(r"lazo cerrado (todo $<0$)", fontsize=8)
axin.set_xlabel(r"$\max_i\,\max\,\mathrm{Re}\,\lambda$", fontsize=7.5); axin.tick_params(labelsize=7)
save(fig, "fig2_estabilidad")

# 3 — Decay-rate alpha logrado (hallazgo clave)
fig, ax = plt.subplots(figsize=(7.2, 3.0))
a = df.alpha_ach.clip(1e-5, None)
ax.hist(a, bins=np.logspace(-4, 1.1, 60), color=C["acc"], alpha=.85)
ax.set_xscale("log")
med = df.alpha_ach.median()
ax.axvline(med, color="k", ls="--", lw=1, label=f"mediana={med:.4f}")
ax.axvline(1e-2, color=C["o"], ls=":", lw=1.2, label="umbral $10^{-2}$")
ax.set_xlabel(r"decay-rate logrado  $\alpha=-\max_i\,\max\,\mathrm{Re}\,\lambda(A_i+B_iK)$")
ax.set_ylabel("nº de sistemas")
ax.set_title(r"El 84.9% queda casi marginal ($\alpha<10^{-2}$)")
ax.legend()
save(fig, "fig3_decay")

# 4 — Plano complejo: el control refleja el espectro al semiplano izquierdo
rng = np.random.default_rng(0)
def _samp(re, im, n=45000):
    i = rng.choice(len(re), min(n, len(re)), replace=False); return re[i], im[i]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for key, c, lab in [("ol", C["ol"], "lazo abierto $A_i$ (inestable)"),
                    ("cl", C["cl"], "lazo cerrado $A_i+B_iK$ (estable)")]:
    re, im = _samp(A[f"{key}_re"], A[f"{key}_im"])
    ax.scatter(re, im, s=2, alpha=.10, color=c, ec="none", rasterized=True, label=lab)
ax.set_xscale("symlog", linthresh=1); ax.set_xlim(-200, 300)
ax.axvline(0, color="k", lw=1.1, ls="--")
ax.set_xlabel(r"$\mathrm{Re}\,\lambda$ (escala symlog)"); ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
ax.set_title("El control refleja el espectro al semiplano izquierdo")
lg = ax.legend(loc="upper left", framealpha=.9, markerscale=4)
for h in lg.legend_handles: h.set_alpha(1)
save(fig, "fig4_plano_complejo")

# 5 — Esfuerzo de control y normas por orden
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
for ax, col, t, c in [(axs[0], "normK2", r"$\|K\|_2$", C["g"]),
                      (axs[1], "normA2_max", r"$\max_i\|A_i\|_2$", C["ol"]),
                      (axs[2], "normB2_max", r"$\max_i\|B_i\|_2$", C["cl"])]:
    data = [df[df.nx == k][col].values for k in sorted(df.nx.unique())]
    bp = ax.boxplot(data, tick_labels=sorted(df.nx.unique()), showfliers=False, patch_artist=True)
    for box in bp["boxes"]: box.set(facecolor=c, alpha=.55)
    ax.set_xlabel("$n_x$"); ax.set_title(t)
fig.suptitle("Magnitudes por orden del sistema", y=1.02)
save(fig, "fig5_normas")

# 6 — Dispersión del politopo vs N
fig, ax = plt.subplots(figsize=(7.2, 3.0))
data = [df[df.N == n].dispA.values for n in sorted(df.N.unique())]
bp = ax.boxplot(data, tick_labels=sorted(df.N.unique()), showfliers=False, patch_artist=True)
for box in bp["boxes"]: box.set(facecolor=C["g"], alpha=.55)
ax.set_xlabel("$N$ (nº de vértices)")
ax.set_ylabel(r"dispersión $\max_i\|A_i-\bar A\|_2$")
ax.set_title("Tamaño del politopo según nº de vértices")
save(fig, "fig6_dispersion")

# 7 — Condicionamiento de A
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.hist(np.log10(df.condA_max), bins=60, color=C["o"], alpha=.85)
ax.axvline(4, color="k", ls="--", lw=1, label=r"$\kappa=10^4$")
ax.set_xlabel(r"$\log_{10}\,\max_i\,\kappa(A_i)$"); ax.set_ylabel("nº de sistemas")
ax.set_title(rf"Condicionamiento de $A$ (cola: {(df.condA_max>1e4).mean()*100:.2f}% mal condicionados)")
ax.legend()
save(fig, "fig7_condicionamiento")

# 8 — Correlaciones entre descriptores
fig, ax = plt.subplots(figsize=(6.2, 5.2))
cols = ["nx", "N", "absc_ol_max", "normA2_max", "normB2_max", "dispA", "condA_max", "alpha_ach", "normK2"]
lab = ["$n_x$", "$N$", "absc OL", r"$\|A\|$", r"$\|B\|$", "dispA", r"$\kappa(A)$", r"$\alpha$", r"$\|K\|$"]
Cm = df[cols].corr(method="spearman").values
im = ax.imshow(Cm, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right")
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
for (i, j), v in np.ndenumerate(Cm):
    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
            color="white" if abs(v) > 0.55 else "black")
ax.grid(False); fig.colorbar(im, ax=ax, shrink=.8, label="Spearman $\\rho$")
ax.set_title("Correlaciones entre descriptores")
save(fig, "fig8_correlaciones")
print("FIGURAS EN:", OUT)
