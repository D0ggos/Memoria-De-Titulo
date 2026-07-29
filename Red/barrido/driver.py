"""
driver.py  —  CLI + multiproceso a nivel de corrida (Parte D)
=============================================================
Reparte las corridas de UNA etapa (o de TODAS, --stage ALL) en procesos: una corrida =
una config completa; modelos chicos, CPU float64. Cada worker fija torch a 1 hilo para no
sobre-suscribir los cores. Cada corrida escribe su propio shard (results/<run_id>,
loss/<run_id>, meta/<run_id>) — sin contencion; el analisis hace glob de todos los shards.

Robustez para corridas largas (dias):
  - RESUME (por defecto): salta corridas cuyo shard meta/<run_id> ya existe. Si el proceso
    se cae, se relanza el MISMO comando y retoma donde iba. --fresh lo desactiva.
  - LOG de progreso: cada corrida completada se anota (con timestamp y ETA) en
    <out>/progreso.log ademas de stdout. Ver avances en vivo con `barrido.estado`.

Uso:
  python -m barrido.driver --stage E0 --workers 4
  python -m barrido.driver --stage ALL --workers 14        # barrido completo (E1-E4), retomable
  python -m barrido.driver --stage E1 --limit 4            # subconjunto para probar
  python -m barrido.driver --stage E1 --cvxpy              # activa techo CVXPY en A1/A2 (lento)
  python -m barrido.driver --stage ALL --fresh             # ignora lo ya hecho y re-corre todo

E5 (OOD) NO se corre aqui: es solo-evaluacion de los modelos ya entrenados (ver
barrido/ood_eval.py y Parte E).
"""
import argparse
import os
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # .../Red
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                             # load_vertices busca el .mat aqui

try:                                                       # consola Windows (cp1252) -> UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch                                               # noqa: E402

from barrido.config import build_stage                     # noqa: E402
from barrido.run import run_one                             # noqa: E402

DEFAULT_OUT = ROOT / "analisis" / "resultados" / "barrido"
STAGE_ALL = ["E1", "E2", "E3", "E4"]                        # orden del barrido completo


_DTYPES = {"float64": torch.float64, "float32": torch.float32}


def _init_worker(dtype_str="float64", threads=1):
    torch.set_num_threads(threads)                        # 1 hilo/corrida en CPU; mas si GPU (1 proc)
    torch.set_default_dtype(_DTYPES[dtype_str])


def _run_cfg(cfg, out_root, cvxpy, device="cpu", dtype_str="float64"):
    if cvxpy:
        cfg = cfg.__class__(**{**cfg.__dict__, "cvxpy_ceiling": True})
    return run_one(cfg, out_root, device=device, dtype=_DTYPES[dtype_str])


def _is_done(out_root, cfg):
    """Una corrida esta COMPLETA si su shard meta existe (meta se escribe al final de
    run_one, tras results y loss). Sirve para el resume."""
    rid = cfg.run_id()
    meta = Path(out_root) / "meta"
    return (meta / f"{rid}.parquet").exists() or (meta / f"{rid}.csv").exists()


def _log(progress_file, msg):
    """Imprime en stdout y agrega al log de progreso (append, con flush)."""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    try:
        with open(progress_file, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass


def _fmt_eta(seconds):
    if seconds != seconds or seconds < 0:                  # NaN o negativo
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def run_stage(stage, out_base, workers, device, dtype_str, threads, cvxpy,
              limit, resume, progress_file, session):
    """Corre UNA etapa. `session` es un dict mutable para llevar contadores/tiempos entre
    etapas (para el ETA global cuando --stage ALL)."""
    out_root = Path(out_base) / stage
    configs = build_stage(stage)
    if limit:
        configs = configs[:limit]

    total_stage = len(configs)
    if resume:
        pending = [c for c in configs if not _is_done(out_root, c)]
        skipped = total_stage - len(pending)
    else:
        pending, skipped = configs, 0

    _log(progress_file, f"=== Etapa {stage}: {total_stage} corridas, {skipped} ya hechas, "
                        f"{len(pending)} por correr -> {out_root} (workers={workers}) ===")
    if not pending:
        _log(progress_file, f"=== Etapa {stage} ya completa. Nada que hacer. ===")
        return

    worker = partial(_run_cfg, out_root=str(out_root), cvxpy=cvxpy,
                     device=device, dtype_str=dtype_str)

    def _handle(i_local, r):
        session["done"] += 1
        elapsed = time.perf_counter() - session["t0"]
        rate = elapsed / max(session["done"], 1)          # s por corrida (de esta sesion)
        remaining = session["total_pending"] - session["done"]
        eta = rate * remaining
        tag = "ok   " if r["status"] == "ok" else "ERROR"
        extra = ""
        if r["status"] == "ok" and "t_train_s" in r:
            extra = f"  t_train={r['t_train_s']:.0f}s"
        _log(progress_file,
             f"[{stage} {i_local}/{len(pending)} | global {session['done']}/{session['total_pending']}] "
             f"{tag}  {r['run_id']}{extra}  (ETA {_fmt_eta(eta)})")
        if r["status"] != "ok":
            session["errors"].append(r)

    if workers == 1:
        _init_worker(dtype_str, threads)
        for i, cfg in enumerate(pending, 1):
            _handle(i, worker(cfg))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")                      # Windows-safe
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(dtype_str, threads)) as pool:
            for i, r in enumerate(pool.imap_unordered(worker, pending), 1):
                _handle(i, r)


def main():
    ap = argparse.ArgumentParser(description="Barrido experimental LMI-Net por etapas.")
    ap.add_argument("--stage", required=True,
                    help="E0 | E1 | E2 | E3 | E4 | ALL (=E1,E2,E3,E4) | lista 'E1,E3'")
    ap.add_argument("--workers", type=int, default=None,
                    help="procesos en paralelo. Defecto: CPU->cpu_count-1, GPU->1")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu",
                    help="donde correr los modelos. Defecto CPU: este barrido es CPU-bound "
                         "(matrices chicas); la GPU va ~5x mas lenta. 'auto' usa CUDA si hay.")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                    help="precision. float64 preserva la numerica original; "
                         "float32 es mucho mas rapido en GPU (puede cambiar resultados)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="usar solo las primeras N corridas/etapa")
    ap.add_argument("--cvxpy", action="store_true", help="activar techo CVXPY en A1/A2")
    ap.add_argument("--fresh", action="store_true",
                    help="re-corre TODO ignorando shards existentes (desactiva el resume)")
    args = ap.parse_args()

    # --- resolver stages ---
    if args.stage.upper() == "ALL":
        stages = STAGE_ALL
    else:
        stages = [s.strip() for s in args.stage.split(",") if s.strip()]

    # --- resolver device ---
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("!! --device cuda pero torch.cuda.is_available()=False. Usando CPU.")
        device = "cpu"

    # --- resolver workers ---
    # En GPU una sola GPU se comparte mal entre procesos (cada uno abre su propio
    # contexto CUDA y compite por los 6 GB); por defecto 1 worker en GPU.
    if args.workers is None:
        workers = 1 if device == "cuda" else max(1, (os.cpu_count() or 2) - 1)
    else:
        workers = args.workers
    if device == "cuda" and workers > 1:
        print(f"!! Aviso: {workers} workers sobre UNA GPU. Comparten VRAM/contexto y "
              f"puede dar OOM o ir mas lento. Considera --workers 1.")
    # threads de CPU por worker: 1 si hay paralelismo por proceso; todos si GPU/1-proc.
    threads = (os.cpu_count() or 1) if (device == "cuda" and workers == 1) else 1

    resume = not args.fresh
    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)
    progress_file = out_base / "progreso.log"

    # --- contar pendientes totales (para el ETA global) ---
    total_pending = 0
    for st in stages:
        cfgs = build_stage(st)
        if args.limit:
            cfgs = cfgs[:args.limit]
        if resume:
            total_pending += sum(1 for c in cfgs if not _is_done(out_base / st, c))
        else:
            total_pending += len(cfgs)

    dev_str = f"{device} ({torch.cuda.get_device_name(0)})" if device == "cuda" else device
    _log(progress_file, "#" * 70)
    _log(progress_file, f"BARRIDO stages={stages}  device={dev_str}  dtype={args.dtype}  "
                        f"workers={workers}  resume={resume}")
    _log(progress_file, f"Corridas pendientes en esta sesion: {total_pending}")

    session = {"t0": time.perf_counter(), "done": 0,
               "total_pending": max(total_pending, 1), "errors": []}

    for st in stages:
        run_stage(st, out_base, workers, device, args.dtype, threads, args.cvxpy,
                  args.limit, resume, progress_file, session)

    dt = time.perf_counter() - session["t0"]
    n_err = len(session["errors"])
    _log(progress_file, f"=== FIN. {session['done']} corridas en {_fmt_eta(dt)}. "
                        f"{n_err} con error. ===")
    for r in session["errors"][:20]:
        _log(progress_file, f"  ERROR {r['run_id']}: {r.get('error')}")
    if not n_err and stages != ["E0"]:
        _log(progress_file, "Genera el reporte final con:  python -m barrido.reporte")


if __name__ == "__main__":
    main()
