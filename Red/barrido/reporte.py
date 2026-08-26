"""
reporte.py  —  reporte .md con las MEJORES configuraciones del barrido (Parte F)
================================================================================
Lee los shards CRUDOS de results/ de todas las etapas y produce un Markdown con las
mejores combinaciones de hiperparametros. NO recomputa nada: solo agrega (promediar en
el analisis SI se permite; la regla de "no promediar" es para los CSV crudos).

Metrica principal (convencion del proyecto): **% estabilizado** en el anillo **A2**
(estructural, generalizacion zero-shot a topologias no vistas) en la epoca final y al
presupuesto de inferencia mas alto (dr_eval=8000). Se promedia sobre semillas y sistemas.
Se reporta tambien A1 (visto), la tasa de decaimiento (decay), la eficiencia (iters_min) y
el condicionamiento kappa(Q).

Uso:
  python -m barrido.reporte                       # lee todas las etapas, escribe REPORTE_BARRIDO.md
  python -m barrido.reporte --stages E1,E4 --out mi_reporte.md
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402

DEFAULT_BASE = ROOT / "analisis" / "resultados" / "barrido"
ALL_STAGES = ["E1", "E2", "E3", "E4", "E7", "E8", "E9"]
DUEL_ARCHS = ["actuadores", "vertices", "vanilla"]        # orden para el duelo arch x n_x
# Ejes de config para el ranking. NO se incluye "stage": la config base se re-entrena
# identica en varias etapas (E1/E2/E4) y agrupar por stage la mostraria como filas
# duplicadas. Sin stage, cada combinacion de hiperparametros aparece una sola vez
# (promediando sus repeticiones entre etapas y semillas).
KEYS = ["loss", "arch", "backprop", "dr_train", "alpha", "n_x", "train_size"]
REF_DR_EVAL = 8000                                        # presupuesto de inferencia de referencia
REF_EPOCH = 400                                           # epoca final del barrido


# --------------------------- IO robusto ------------------------------------
def read_results(base, stages):
    """Concatena results/*.parquet|csv de las etapas dadas. DataFrame vacio si no hay."""
    frames = []
    for st in stages:
        folder = Path(base) / st / "results"
        if not folder.exists():
            continue
        for p in sorted(list(folder.glob("*.parquet")) + list(folder.glob("*.csv"))):
            try:
                frames.append(pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p))
            except Exception as e:                          # noqa: BLE001
                print(f"  aviso: no pude leer {p.name}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Los shards anteriores a E7/E8 no traen columnas sigma/epsilon: se rellenan con el
    # default que esas corridas usaron de hecho (epsilon depende de la arquitectura), para
    # que puedan compararse contra el barrido de sensibilidad sin quedar como NaN.
    if "sigma" not in df.columns:
        df["sigma"] = np.nan
    if "epsilon" not in df.columns:
        df["epsilon"] = np.nan
    df["sigma"] = df["sigma"].fillna(0.01)
    df["epsilon"] = df["epsilon"].fillna(
        pd.Series(np.where(df["arch"] == "vanilla", 1e-3, 1e-5), index=df.index))
    return df


def count_meta(base, stages):
    """Cuenta shards meta (=corridas completas) por etapa."""
    out = {}
    for st in stages:
        folder = Path(base) / st / "meta"
        n = len(list(folder.glob("*.parquet")) + list(folder.glob("*.csv"))) if folder.exists() else 0
        out[st] = n
    return out


# --------------------------- Agregacion ------------------------------------
def _pick(df, epoch, dr_eval):
    """Filtra a (epoch, dr_eval), con fallback al max disponible si faltan (datos parciales)."""
    ep = epoch if (df.epoch == epoch).any() else df.epoch.max()
    dre = dr_eval if (df.dr_eval == dr_eval).any() else df.dr_eval.max()
    return df[(df.epoch == ep) & (df.dr_eval == dre)], ep, dre


def leaderboard(df, ring, keys=KEYS, epoch=REF_EPOCH, dr_eval=REF_DR_EVAL):
    """Tabla por-config (promedio sobre semillas y sistemas) para un anillo, ordenada por
    % estabilizado descendente."""
    sub = df[df.ring == ring]
    if sub.empty:
        return pd.DataFrame(), (None, None)
    sub, ep, dre = _pick(sub, epoch, dr_eval)
    keys = [k for k in keys if k in sub.columns]
    g = sub.groupby(keys, dropna=False)
    agg = g.agg(
        stable_pct=("stable", "mean"),
        decay_pct=("decay", "mean"),
        iters_min_med=("iters_min", "median"),
        kappa_med=("kappa_Q", "median"),
        margin_med=("margin", "median"),
        n_sys=("stable", "size"),
        seeds=("seed", "nunique"),
    ).reset_index()
    agg["stable_pct"] *= 100.0
    agg["decay_pct"] *= 100.0
    agg = agg.sort_values(["stable_pct", "decay_pct", "margin_med"],
                          ascending=[False, False, False]).reset_index(drop=True)
    return agg, (ep, dre)


# --------------------------- Render markdown -------------------------------
def _fmt(v, col):
    if isinstance(v, float) and (v != v):                 # NaN
        return "—"
    if col in ("stable_pct", "decay_pct"):
        return f"{v:.1f}"
    if col in ("iters_min_med", "kappa_med", "margin_med"):
        return f"{v:.3g}" if isinstance(v, float) else str(v)
    if col == "dr_train":
        return "impl" if (v != v) else f"{int(v)}"
    if col in ("n_x", "train_size", "n_sys", "seeds"):
        return "—" if (isinstance(v, float) and v != v) else f"{int(v)}"
    if col == "alpha":
        return f"{v:g}"
    return str(v)


def md_table(df, cols, headers=None, max_rows=None):
    if df.empty:
        return "_(sin datos)_\n"
    headers = headers or cols
    rows = df.head(max_rows) if max_rows else df
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for _, r in rows.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], c) for c in cols) + " |")
    return "\n".join(out) + "\n"


LB_COLS = ["loss", "arch", "backprop", "dr_train", "alpha", "n_x", "train_size",
           "stable_pct", "decay_pct", "iters_min_med", "kappa_med", "seeds"]
LB_HEAD = ["pérdida", "arqu.", "backprop", "dr_tr", "α", "n_x", "sz",
           "A? %estab", "%decay", "iters_min", "κ(Q)", "seeds"]


def axis_summary(df, ring, axis, epoch=REF_EPOCH, dr_eval=REF_DR_EVAL):
    """% estabilizado promedio por valor de un eje (marginando el resto)."""
    sub = df[df.ring == ring]
    if sub.empty or axis not in sub.columns:
        return pd.DataFrame()
    sub, _, _ = _pick(sub, epoch, dr_eval)
    g = sub.groupby(axis, dropna=False).agg(
        stable_pct=("stable", "mean"), decay_pct=("decay", "mean"),
        iters_min_med=("iters_min", "median"), n_sys=("stable", "size")).reset_index()
    g["stable_pct"] *= 100.0
    g["decay_pct"] *= 100.0
    return g.sort_values("stable_pct", ascending=False).reset_index(drop=True)


# --------------------------- Duelo arquitectura x n_x ----------------------
def architecture_duel(df, ring, epoch=REF_EPOCH, dr_eval=REF_DR_EVAL):
    """Pivot n_x x arquitectura de % estabilizado en la CONFIG BASE controlada
    (unrolling/dr30/alpha0.01/sz150), promedio sobre perdidas y semillas. Asi la unica
    diferencia entre celdas es arquitectura y orden (head-to-head justo). None si no hay."""
    s = df[(df.ring == ring) & (df.epoch == epoch) & (df.dr_eval == dr_eval) &
           (df.backprop == "unrolling") & (df.dr_train == 30) &
           (df.alpha == 0.01) & (df.train_size == 150)]
    if s.empty:
        return None
    piv = (s.groupby(["n_x", "arch"]).stable.mean() * 100).unstack("arch")
    cols = [a for a in DUEL_ARCHS if a in piv.columns]
    return piv.reindex(columns=cols).sort_index().round(1)


def _pivot_md(piv, row_name="n_x"):
    """Renderiza un pivot (indice=n_x, columnas=arquitectura) como tabla markdown."""
    heads = [row_name] + list(piv.columns)
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in heads) + "|"]
    for idx, r in piv.iterrows():
        cells = [str(int(idx))] + [("—" if pd.isna(v) else f"{v:.1f}") for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# --------------------------- Main ------------------------------------------
def build_report(base, stages):
    df = read_results(base, stages)
    meta = count_meta(base, stages)
    expected = {}
    try:
        from barrido.config import build_stage
        for st in stages:
            expected[st] = len(build_stage(st))
    except Exception:
        expected = {st: None for st in stages}

    L = []
    L.append(f"# Reporte del barrido LMI-Net\n")
    L.append(f"_Generado: {datetime.now():%Y-%m-%d %H:%M:%S}_\n")

    # --- cobertura ---
    L.append("## Cobertura\n")
    cov = pd.DataFrame([{"etapa": st, "completas": meta.get(st, 0),
                         "esperadas": expected.get(st), }
                        for st in stages])
    if not cov.empty:
        cov["esperadas"] = cov["esperadas"].apply(lambda v: "?" if v is None else int(v))
        L.append(md_table(cov.assign(pct=[
            "—" if not e or e == "?" else f"{100*c/e:.0f}%"
            for c, e in zip(cov.completas, [expected.get(st) for st in stages])]),
            ["etapa", "completas", "esperadas", "pct"],
            ["etapa", "completas", "esperadas", "%"]))
    if df.empty:
        L.append("\n> ⚠️ Aún no hay resultados (`results/`) que analizar. "
                 "Corre el barrido o espera a que terminen las primeras corridas.\n")
        return "\n".join(L)

    total_sys = len(df)
    L.append(f"\nFilas crudas analizadas: **{total_sys:,}** "
             f"(una por sistema × anillo × época × dr_eval). "
             f"Métrica de referencia: **% estabilizado en A2** (época {REF_EPOCH}, "
             f"dr_eval={REF_DR_EVAL}), promedio sobre semillas.\n")

    # --- headline: mejor config global en A2 ---
    lbA2, (ep2, dre2) = leaderboard(df, "A2")
    lbA1, (ep1, dre1) = leaderboard(df, "A1")
    L.append("## 🏆 Mejor configuración (generalización A2)\n")
    if not lbA2.empty:
        best = lbA2.iloc[0]
        dr_txt = "implícito" if (best["dr_train"] != best["dr_train"]) else f"dr_train={int(best['dr_train'])}"
        L.append(f"A la época {ep2}, dr_eval={dre2}, la mejor combinación en el anillo "
                 f"estructural A2 es:\n")
        L.append(f"- **pérdida = `{best['loss']}`**, arquitectura = `{best['arch']}`, "
                 f"backprop = `{best['backprop']}` ({dr_txt}), α = {best['alpha']:g}, "
                 f"n_x = {int(best['n_x'])}, train_size = {int(best['train_size'])}\n")
        L.append(f"- **A2: {best['stable_pct']:.1f}% estabilizado**, {best['decay_pct']:.1f}% "
                 f"cumple decay; iters_min mediano = {_fmt(best['iters_min_med'],'iters_min_med')}, "
                 f"κ(Q) mediano = {_fmt(best['kappa_med'],'kappa_med')} "
                 f"(sobre {int(best['seeds'])} semillas)\n")

    # --- leaderboards A2 y A1 ---
    L.append("\n## Ranking global de configuraciones\n")
    L.append(f"### Anillo A2 — estructural / zero-shot (época {ep2}, dr_eval={dre2})\n")
    L.append(md_table(lbA2, LB_COLS, [h.replace("A? ", "A2 ") for h in LB_HEAD], max_rows=15))
    L.append(f"\n### Anillo A1 — visto (época {ep1}, dr_eval={dre1})\n")
    L.append(md_table(lbA1, LB_COLS, [h.replace("A? ", "A1 ") for h in LB_HEAD], max_rows=15))

    # --- ganadores por eje ---
    L.append("\n## Efecto de cada eje (A2, marginando el resto)\n")
    axes = [("loss", "Función de pérdida"), ("backprop", "Backpropagation"),
            ("arch", "Arquitectura"), ("alpha", "α (decay prescrito)"),
            ("dr_train", "Presupuesto DR de entrenamiento"),
            ("n_x", "Orden del sistema n_x"), ("train_size", "Tamaño de entrenamiento")]
    for ax_col, ax_name in axes:
        g = axis_summary(df, "A2", ax_col)
        if g.empty or g[ax_col].nunique() < 2:
            continue
        L.append(f"\n**{ax_name}** — mejor: `{_fmt(g.iloc[0][ax_col], ax_col)}` "
                 f"({g.iloc[0]['stable_pct']:.1f}% A2)\n")
        L.append(md_table(g, [ax_col, "stable_pct", "decay_pct", "iters_min_med", "n_sys"],
                          [ax_name, "%estab A2", "%decay", "iters_min", "n_sys"]))

    # --- eficiencia: mejores iters_min entre configs con A2>=90% ---
    L.append("\n## Eficiencia (menor iters_min con A2 ≥ 90% estabilizado)\n")
    if not lbA2.empty:
        eff = lbA2[lbA2["stable_pct"] >= 90.0].copy()
        eff = eff.sort_values("iters_min_med").reset_index(drop=True)
        if eff.empty:
            L.append("_(ninguna config alcanza 90% en A2 todavía)_\n")
        else:
            L.append(md_table(eff, LB_COLS, [h.replace("A? ", "A2 ") for h in LB_HEAD], max_rows=10))

    # --- duelo de arquitecturas x n_x (E4 aporta vertices/vanilla; E2, actuadores) ---
    dA2 = architecture_duel(df, "A2"); dA1 = architecture_duel(df, "A1")
    if dA2 is not None or dA1 is not None:
        L.append("\n## Duelo de arquitecturas × n_x\n")
        L.append("Config base controlada (`unrolling / dr30 / α=0.01 / sz150`); "
                 "% estabilizado promedio sobre las 6 pérdidas × 2 semillas. Única diferencia "
                 "entre celdas: arquitectura y orden. `vanilla` no es invariante → no tiene A2.\n")
        if dA2 is not None:
            L.append("\n**A2 — generalización zero-shot (topología N=5 / n_u=2 no vista)**\n")
            L.append(_pivot_md(dA2))
        if dA1 is not None:
            L.append("\n**A1 — sistemas vistos**\n")
            L.append(_pivot_md(dA1))
        L.append("\n_Lectura: en A2, `actuadores` gana en todos los órdenes (su ventaja sobre "
                 "`vertices` crece con n_x); en A1, `vanilla` domina pero no generaliza. "
                 "Cobertura: `actuadores` de E1/E2, `vertices`/`vanilla` de E4._\n")

    L.append("\n---\n_Métricas por-sistema promediadas en el análisis. "
             "A2 = topologías retenidas (N=5, n_u=2) nunca vistas en entrenamiento._\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="Reporte .md de las mejores configs del barrido.")
    ap.add_argument("--base", default=str(DEFAULT_BASE), help="carpeta raíz del barrido")
    ap.add_argument("--stages", default="ALL", help="ALL o lista 'E1,E4'")
    ap.add_argument("--out", default=None, help="ruta del .md (defecto <base>/REPORTE_BARRIDO.md)")
    args = ap.parse_args()

    stages = ALL_STAGES if args.stages.upper() == "ALL" else \
        [s.strip() for s in args.stages.split(",") if s.strip()]
    out = Path(args.out) if args.out else Path(args.base) / "REPORTE_BARRIDO.md"

    md = build_report(args.base, stages)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(f"Reporte escrito en {out}")


if __name__ == "__main__":
    main()
