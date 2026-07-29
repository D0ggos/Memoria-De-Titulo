"""
run.py  —  entrena UNA corrida con checkpoints de epoca y evalua los anillos (Parte D)
======================================================================================
Una corrida = una RunConfig. Se entrena una sola vez hasta cfg.epochs; en cada checkpoint
de epoca se (a) guarda el modelo y (b) se evalua sobre A1 (visto) y A2 (estructural) con el
ladder de dr_eval (una pasada con checkpoints). TODO se escribe en crudo (una fila por
sistema x anillo x epoca x dr_eval); el analisis y las figuras van aparte (Parte F).

Semilla = columna, nunca se promedia aqui. dr_eval y epoca son checkpoints: NO multiplican
el nº de entrenamientos.
"""
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import psutil
    _PROC = psutil.Process()
except Exception:                                          # psutil opcional
    _PROC = None


def _rss_mb():
    return _PROC.memory_info().rss / 1e6 if _PROC is not None else float("nan")

from entrenamiento.training import make_batches, get_loss_fn
from red.vanilla import LMINetVanilla
from red.vertices import LMINetVertices
from red.actuators import LMINetActuators

from .config import BASE_N
from .data import make_datasets
from .evaluation import evaluate_ring


def build_model(cfg, dtype=torch.float64):
    """Instancia la arquitectura de la corrida con la precision `dtype` (float64 por
    defecto). Para implicit, dr_iters solo se usa como presupuesto por defecto (el forward
    implicito corre hasta implicit_max_iters)."""
    dr = cfg.dr_train if cfg.dr_train is not None else 500
    if cfg.arch == "actuadores":
        model = LMINetActuators(n=cfg.n_x, alpha=cfg.alpha, dr_iters=dr, backprop=cfg.backprop)
    elif cfg.arch == "vertices":
        model = LMINetVertices(n=cfg.n_x, m=1, alpha=cfg.alpha, dr_iters=dr, backprop=cfg.backprop)
    elif cfg.arch == "vanilla":
        model = LMINetVanilla(n=cfg.n_x, m=1, N=BASE_N, alpha=cfg.alpha, dr_iters=dr,
                              backprop=cfg.backprop)
    else:
        raise ValueError(f"arquitectura desconocida: {cfg.arch}")
    return model.to(dtype)


def save_table(df, path_noext):
    """Escribe parquet si hay engine; si no, CSV. Devuelve la ruta escrita. Part F busca
    ambos formatos."""
    p = Path(path_noext)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = p.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        return out
    except Exception:
        out = p.with_suffix(".csv")
        df.to_csv(out, index=False)
        return out


def _train_epoch(model, batches, loss_fn, opt):
    run_, nb = 0.0, 0
    for A, B in batches:
        opt.zero_grad()
        Q, Y = model(A, B)
        loss = loss_fn(Q, Y, A, B)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_ += loss.item(); nb += 1
    return run_ / max(nb, 1)


def run_one(cfg, out_root, lr=1e-3, batch=16, device="cpu", dtype=torch.float64):
    """Ejecuta una corrida completa. Devuelve un dict-resumen. Robusto a fallos: captura la
    excepcion, la escribe en un .err y sigue (para no tumbar el Pool del barrido).
    `device`/`dtype`: donde corre el modelo (CPU/'cuda') y con que precision."""
    out_root = Path(out_root)
    rid = cfg.run_id()
    try:
        torch.set_default_dtype(dtype)
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

        train_by_cell, a1, a2 = make_datasets(cfg.arch, cfg.n_x, cfg.train_size)
        if not train_by_cell:
            raise RuntimeError(f"sin celdas de entrenamiento para arch={cfg.arch} n_x={cfg.n_x}")

        model = build_model(cfg, dtype=dtype).to(device)
        loss_fn = get_loss_fn(cfg.loss, model)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.default_rng(cfg.seed)

        ckpts = set(int(e) for e in cfg.epoch_ckpts)
        cfg_cols = cfg.as_row()
        result_rows, loss_rows = [], []
        t_train, mem_peak = 0.0, _rss_mb()
        model.train()
        for ep in range(1, cfg.epochs + 1):
            batches = []
            for _cell, items in train_by_cell.items():
                batches += make_batches(items, batch=batch, shuffle=True, rng=rng,
                                        device=device, dtype=dtype)
            rng.shuffle(batches)
            t0 = time.perf_counter()
            tr_loss = _train_epoch(model, batches, loss_fn, opt)
            t_train += time.perf_counter() - t0
            mem_peak = max(mem_peak, _rss_mb())
            loss_rows.append({**cfg_cols, "epoch": ep, "train_loss": tr_loss})

            if ep in ckpts:
                mdl_dir = out_root / "models" / rid
                mdl_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"cfg": cfg_cols, "epoch": ep, "state_dict": model.state_dict()},
                           mdl_dir / f"epoch_{ep}.pt")
                model.eval()
                for ring_name, ring in (("A1", a1), ("A2", a2)):
                    if not ring:
                        continue
                    rrows = evaluate_ring(model, ring, cfg.dr_eval, ring_name,
                                          cvxpy=cfg.cvxpy_ceiling, cvxpy_max=cfg.cvxpy_max_systems,
                                          device=device, dtype=dtype)
                    for row in rrows:
                        row.update(cfg_cols); row["epoch"] = ep; row["train_loss"] = tr_loss
                    result_rows.extend(rrows)
                model.train()

        save_table(pd.DataFrame(result_rows), out_root / "results" / rid)
        save_table(pd.DataFrame(loss_rows), out_root / "loss" / rid)
        save_table(pd.DataFrame([{**cfg_cols, "t_train_s": t_train, "mem_peak_mb": mem_peak,
                                  "n_train_systems": sum(len(v) for v in train_by_cell.values())}]),
                   out_root / "meta" / rid)
        return dict(run_id=rid, status="ok", n_rows=len(result_rows), epochs=cfg.epochs,
                    t_train_s=t_train, mem_peak_mb=mem_peak)

    except Exception as e:                                      # noqa: BLE001
        err_dir = out_root / "errors"; err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / f"{rid}.err").write_text(f"{e}\n\n{traceback.format_exc()}", encoding="utf-8")
        return dict(run_id=rid, status="error", error=str(e))
