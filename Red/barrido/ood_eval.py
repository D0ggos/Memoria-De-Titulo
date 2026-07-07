"""
ood_eval.py  —  evaluacion OOD y mecanismo de fallo del profesor (Parte E / E5)
===============================================================================
E5 NO reentrena: carga los modelos ya entrenados (checkpoints del barrido) y los evalua
sobre los bancos OOD (pole-shift y politopo del profesor) con el ladder de dr_eval en UNA
pasada con checkpoints. Como TODAS las perdidas se entrenaron, el analisis OOD las compara
todas.

METRICA OOD: iters_min = menor presupuesto DR que estabiliza. CAVEAT (obligatorio): se
reporta junto a estab_en_max (la trayectoria de DR NO es monotona; iters_min solo es
concluyente si estab_en_max=True).

MECANISMO DE FALLO (clave para la tesis). Hipotesis a validar/refutar: el encoder entrenado
con perdidas que premian Q pequeña extrapola OOD apuntando ŷ hacia una esquina casi-singular
donde los conjuntos C1 (afin) y C2 (cono PSD) se cortan casi TANGENCIALMENTE -> DR converge
lentisimo -> K=YQ⁻¹ explota a presupuesto corto. Prediccion distintiva: el APLANAMIENTO del
residual de DR vs. k aparece SOLO para ŷ de la red, no para ŷ aleatorio (que aterriza en un
punto generico y sale rapido). Por eso se instrumenta, por δ y por iteracion k:
  - peor autovalor de lazo cerrado max_i Re λ(A_i+B_iK) vs k, red vs aleatorio (mediana de
    n_random semillas);
  - residual de DR vs k (aplanamiento = convergencia lenta por tangencia);
  - λ_min(Q*), κ(Q*) del certificado final vs δ;
  - factibilidad CVXPY por δ (verdad de terreno: los fallos deben ser del metodo);
  - tiempo hasta estabilizar (ms) red vs CVXPY vs δ.
"""
import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
import torch                                           # noqa: E402

from analisis.benchmark import _cvxpy_solve            # noqa: E402
from barrido.config import DR_EVAL                      # noqa: E402
from barrido.run import build_model, save_table         # noqa: E402
from barrido.metrics import certificate_metrics         # noqa: E402
from barrido.evaluation import dr_eval_checkpoints, _iters_min   # noqa: E402
from barrido.ood_banks import (pole_shift_bank, professor_bank,   # noqa: E402
                               professor_system, PROF_DELTAS)

# grilla de k para la traza del mecanismo (densa al inicio, log al final)
K_GRID = sorted(set(list(range(25, 500, 25)) +
                    [500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]))


# --------------------------- carga de modelos ------------------------------
def load_checkpoint(path):
    """Reconstruye el modelo desde un checkpoint del barrido (guarda cfg + state_dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**ck["cfg"])
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    model.double().eval()
    return model, ck["cfg"], int(ck["epoch"])


# --------------------------- traza de DR -----------------------------------
@torch.no_grad()
def dr_trace(model, y_hat, L, c, M_inv, k_grid):
    """Corre DR desde ŷ y en cada k de k_grid devuelve (y*_proyectada, residual_DR).
    residual = ||incremento de DR||_inf en ese paso (=0 solo en el punto fijo)."""
    ks = sorted(set(int(k) for k in k_grid)); kset = set(ks)
    y_k = y_hat.clone()
    x_k = torch.bmm(L, y_k.unsqueeze(-1)).squeeze(-1) + c
    out = {}
    for it in range(1, ks[-1] + 1):
        y_next, x_next = model._dr_step_batch(y_k, x_k, y_hat, L, c, M_inv)
        resid = (x_next - x_k).abs().max().item()
        y_k, x_k = y_next, x_next
        if it in kset:
            out[it] = (model._dr_final_proj(y_hat, y_k, x_k, L, c, M_inv), resid)
    return out


# --------------------------- E5: ladder OOD --------------------------------
@torch.no_grad()
def evaluate_bank_ladder(model, bank, budgets, bank_name, cfg_cols):
    """Evalua un banco (lista de (param, A, B)) sobre el ladder de dr_eval. Una fila por
    (param x dr_eval) con metricas del certificado + iters_min/estab_en_max."""
    rows = []
    for param, A, B in bank:
        Ab, Bb = A.unsqueeze(0), B.unsqueeze(0)                 # batch 1
        y_hat = model(Ab, Bb, return_unconstrained=True)
        L, c, M_inv = model._dr_precompute(Ab, Bb)
        ck = dr_eval_checkpoints(model, y_hat, L, c, M_inv, budgets)
        stable_by_ms = {}
        mets_by_ms = {}
        for it, (y_star, t_s) in ck.items():
            m = certificate_metrics(model, y_star, Ab, Bb, y_hat=y_hat)
            mets_by_ms[it] = (m, t_s); stable_by_ms[it] = m["stable"]
        imin, estab_max = _iters_min(stable_by_ms, list(ck.keys()))
        for it, (m, t_s) in mets_by_ms.items():
            row = dict(bank=bank_name, param=param, dr_eval=it,
                       t_batch_s=t_s, t_unit_ms=t_s * 1e3,
                       iters_min=float(imin[0]), estab_en_max=bool(estab_max[0]))
            for k, arr in m.items():
                row[k] = arr[0].item() if hasattr(arr[0], "item") else arr[0]
            row.update(cfg_cols)
            rows.append(row)
    return rows


# --------------------------- mecanismo del profesor ------------------------
@torch.no_grad()
def professor_mechanism(model, cfg_cols, deltas=PROF_DELTAS, budgets=DR_EVAL,
                        k_grid=K_GRID, n_random=5, rng_seed=0):
    """Instrumentacion del mecanismo de fallo sobre el politopo del profesor. Devuelve
    (trace_rows, summary_rows):
      trace_rows  : por (δ, source∈{red,aleatorio}, k) -> worst, residual, λ_min(Q), κ(Q).
                    'aleatorio' = mediana de n_random semillas (control del solver: DR-puro
                    desde ŷ aleatorio).
      summary_rows: por δ -> iters_min/estab_en_max (red), λ_min(Q*)/κ(Q*) finales,
                    factibilidad CVXPY, tiempo hasta estabilizar red vs CVXPY."""
    trace_rows, summary_rows = [], []
    for d in deltas:
        A, B = professor_system(d)                              # normalizado, (N,n,n)
        Ab, Bb = A.unsqueeze(0), B.unsqueeze(0)
        y_hat = model(Ab, Bb, return_unconstrained=True)
        L, c, M_inv = model._dr_precompute(Ab, Bb)

        # --- traza red ---
        tr_net = dr_trace(model, y_hat, L, c, M_inv, k_grid)
        for k, (y_star, res) in tr_net.items():
            m = certificate_metrics(model, y_star, Ab, Bb)
            trace_rows.append(dict(bank="profesor", delta=d, source="red", k=k,
                                   worst=float(m["worst"][0]), residual=res,
                                   lam_min_Q=float(m["lam_min_Q"][0]),
                                   kappa_Q=float(m["kappa_Q"][0]), **cfg_cols))

        # --- traza aleatoria (mediana de n_random semillas) ---
        dim = y_hat.shape[-1]
        rng = np.random.default_rng(rng_seed)
        acc_w = {k: [] for k in tr_net}; acc_r = {k: [] for k in tr_net}
        for _s in range(n_random):
            yh = torch.from_numpy(rng.standard_normal((1, dim)) * 0.3).to(A.dtype)
            tr = dr_trace(model, yh, L, c, M_inv, k_grid)
            for k, (y_star, res) in tr.items():
                acc_w[k].append(float(certificate_metrics(model, y_star, Ab, Bb)["worst"][0]))
                acc_r[k].append(res)
        for k in tr_net:
            trace_rows.append(dict(bank="profesor", delta=d, source="aleatorio", k=k,
                                   worst=float(np.median(acc_w[k])),
                                   residual=float(np.median(acc_r[k])),
                                   lam_min_Q=np.nan, kappa_Q=np.nan, **cfg_cols))

        # --- resumen por δ (red) sobre el ladder de dr_eval, con tiempos ---
        ck = dr_eval_checkpoints(model, y_hat, L, c, M_inv, budgets)
        stable_by_ms, mets_by_ms = {}, {}
        for it, (y_star, t_s) in ck.items():
            mm = certificate_metrics(model, y_star, Ab, Bb)
            mets_by_ms[it] = (mm, t_s); stable_by_ms[it] = mm["stable"]
        imin, estab_max = _iters_min(stable_by_ms, list(ck.keys()))
        bmax = max(ck.keys())
        t_stab_net = ck[int(imin[0])][1] * 1e3 if not np.isnan(imin[0]) else ck[bmax][1] * 1e3
        m_final = mets_by_ms[bmax][0]
        feas, t_cvx, _, _ = _cvxpy_solve(model, A, B)
        summary_rows.append(dict(
            bank="profesor", delta=d,
            iters_min=float(imin[0]), estab_en_max=bool(estab_max[0]),
            lam_min_Q_final=float(m_final["lam_min_Q"][0]),
            kappa_Q_final=float(m_final["kappa_Q"][0]),
            worst_final=float(m_final["worst"][0]),
            cvxpy_feasible=(None if feas is None else bool(feas)),
            t_stab_red_ms=t_stab_net,
            t_cvxpy_ms=(t_cvx * 1e3 if t_cvx is not None else np.nan),
            **cfg_cols))
    return trace_rows, summary_rows


# --------------------------- driver E5 -------------------------------------
def _bank_for(cfg):
    """Elige el banco aplicable segun el orden del modelo. Solo arquitecturas invariantes
    (actuadores/vertices) pueden aplicarse a un N distinto al de entrenamiento."""
    if cfg.get("arch") == "vanilla":
        return None
    nx = cfg.get("n_x")
    if nx == 3:
        return "pole_shift"
    if nx == 2:
        return "profesor"
    return None


def run_ood(models_root, out_root, budgets=DR_EVAL, mechanism=True, final_only=True,
            n_random=5):
    """Recorre los checkpoints del barrido bajo models_root, evalua cada modelo aplicable
    en su banco y escribe shards crudos. Para el banco del profesor, ademas corre la
    instrumentacion del mecanismo (trace + summary)."""
    models_root = Path(models_root); out_root = Path(out_root)
    ckpts = sorted(models_root.rglob("epoch_*.pt"))
    if final_only:
        # una entrada por run_id: el checkpoint de mayor epoca (el modelo final).
        by_run = {}
        for p in ckpts:
            ep = int(p.stem.split("_")[1])
            if p.parent.name not in by_run or ep > by_run[p.parent.name][0]:
                by_run[p.parent.name] = (ep, p)
        ckpts = [p for _ep, p in by_run.values()]
    print(f"OOD: {len(ckpts)} checkpoints a evaluar bajo {models_root} (final_only={final_only})")
    for path in ckpts:
        model, cfg, epoch = load_checkpoint(path)
        bank_name = _bank_for(cfg)
        if bank_name is None:
            continue
        cfg_cols = {**cfg, "ckpt_epoch": epoch}
        rid = cfg.get("run_id", path.parent.name)
        tag = f"{rid}__ep{epoch}"
        if bank_name == "pole_shift":
            rows = evaluate_bank_ladder(model, pole_shift_bank(), budgets, "pole_shift", cfg_cols)
            save_table(pd.DataFrame(rows), out_root / "ladder" / tag)
        else:  # profesor
            rows = evaluate_bank_ladder(model, professor_bank(), budgets, "profesor", cfg_cols)
            save_table(pd.DataFrame(rows), out_root / "ladder" / tag)
            if mechanism:
                tr, summ = professor_mechanism(model, cfg_cols, budgets=budgets, n_random=n_random)
                save_table(pd.DataFrame(tr), out_root / "mecanismo_trace" / tag)
                save_table(pd.DataFrame(summ), out_root / "mecanismo_summary" / tag)
        print(f"  OK {tag}  banco={bank_name}")


def main():
    ap = argparse.ArgumentParser(description="Evaluacion OOD (E5) de modelos entrenados.")
    ap.add_argument("--models", required=True, help="carpeta con checkpoints (models/ del barrido)")
    ap.add_argument("--out", default=str(ROOT / "analisis" / "resultados" / "barrido" / "OOD"))
    ap.add_argument("--no-mechanism", action="store_true", help="omitir la instrumentacion del profesor")
    ap.add_argument("--all-epochs", action="store_true", help="evaluar todos los checkpoints, no solo el final")
    ap.add_argument("--n-random", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    run_ood(args.models, args.out, mechanism=not args.no_mechanism,
            final_only=not args.all_epochs, n_random=args.n_random)


if __name__ == "__main__":
    main()
