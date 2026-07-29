"""
evaluation.py  —  ladder de dr_eval por checkpoint y evaluacion de un anillo (Parte D)
======================================================================================
El presupuesto de DR en inferencia (dr_eval) se recorre en UNA sola pasada con
checkpoints: se corre DR desde el ŷ del encoder hasta max(ladder) y en cada hito se
proyecta (Pi_{C1}) y se miden las metricas por-sistema. Asi dr_eval NO multiplica el
computo por milestone (solo una pasada).

Reusa el paso de DR de core (_dr_step_batch, _dr_final_proj): NO se reimplementa la
matematica del solver.

CAVEAT del enunciado sobre iters_min: la trayectoria de DR NO es monotona (hay cruces
transitorios), asi que junto a `iters_min` (menor presupuesto que estabiliza) se reporta
`estab_en_max` (estable al presupuesto maximo). iters_min solo es concluyente si
estab_en_max=True.
"""
import time

import numpy as np
import torch

from .data import stack_cell
from .metrics import certificate_metrics, cvxpy_feasible_batch


@torch.no_grad()
def dr_eval_checkpoints(model, y_hat, L, c, M_inv, milestones):
    """Corre DR desde ŷ hasta max(milestones); en cada hito devuelve (y*, t_acumulado_s).
    y* es la proyeccion final Pi_{C1} del estado en ese hito (misma salida que el forward)."""
    ms = set(int(x) for x in milestones)
    y_k = y_hat.clone()
    x_k = torch.bmm(L, y_k.unsqueeze(-1)).squeeze(-1) + c
    out = {}
    t0 = time.perf_counter()
    for it in range(1, max(ms) + 1):
        y_k, x_k = model._dr_step_batch(y_k, x_k, y_hat, L, c, M_inv)
        if it in ms:
            y_star = model._dr_final_proj(y_hat, y_k, x_k, L, c, M_inv)
            out[it] = (y_star, time.perf_counter() - t0)
    return out


def _iters_min(sys_stable_by_ms, milestones):
    """Por sistema: menor milestone con estable=True (o NaN si nunca), y estable@max."""
    ms_sorted = sorted(milestones)
    B = sys_stable_by_ms[ms_sorted[0]].shape[0]
    imin = np.full(B, np.nan)
    for it in ms_sorted:
        st = sys_stable_by_ms[it]
        newly = np.isnan(imin) & st
        imin[newly] = it
    estab_max = sys_stable_by_ms[ms_sorted[-1]]
    return imin, estab_max


@torch.no_grad()
def evaluate_ring(model, ring_cells, milestones, ring_name,
                  cvxpy=False, cvxpy_max=30, device="cpu", dtype=None):
    """Evalua un anillo (dict (m,N)->items) sobre el ladder de dr_eval. Devuelve una lista
    de filas CRUDAS (una por sistema x milestone). Estratificado por celda (m,N)."""
    rows = []
    model.eval()
    for (m, N), items in ring_cells.items():
        if not items:
            continue
        A, B = stack_cell(items, device=device, dtype=dtype)
        y_hat = model(A, B, return_unconstrained=True)         # fija model.N, model.m
        L, c, M_inv = model._dr_precompute(A, B)
        ckpts = dr_eval_checkpoints(model, y_hat, L, c, M_inv, milestones)

        # feasibilidad CVXPY por celda (una vez; verdad de terreno / techo)
        feas_col = None
        if cvxpy:
            fb = cvxpy_feasible_batch(model, A, B, y_hat, max_systems=cvxpy_max)
            feas_col = np.full(A.shape[0], np.nan, dtype=float)
            feas_col[:fb["n"]] = fb["feasible"].astype(float)

        # metricas por milestone + tabla de estabilidad para iters_min
        per_ms = {}
        stable_by_ms = {}
        for it, (y_star, t_s) in ckpts.items():
            mets = certificate_metrics(model, y_star, A, B, y_hat=y_hat)
            per_ms[it] = (mets, t_s)
            stable_by_ms[it] = mets["stable"]
        imin, estab_max = _iters_min(stable_by_ms, list(ckpts.keys()))

        for it, (mets, t_s) in per_ms.items():
            Bsz = mets["worst"].shape[0]
            for b in range(Bsz):
                row = dict(ring=ring_name, m=m, N=N, sys=b, dr_eval=it,
                           t_batch_s=t_s, t_unit_ms=t_s / Bsz * 1e3,
                           iters_min=imin[b], estab_en_max=bool(estab_max[b]))
                for k, arr in mets.items():
                    row[k] = arr[b].item() if hasattr(arr[b], "item") else arr[b]
                if feas_col is not None:
                    row["cvxpy_feasible"] = feas_col[b]
                rows.append(row)
    return rows
