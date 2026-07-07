"""
driver.py  —  CLI + multiproceso a nivel de corrida (Parte D)
=============================================================
Reparte las corridas de UNA etapa en procesos (una corrida = una config completa; modelos
chicos, CPU float64). Cada worker fija torch a 1 hilo para no sobre-suscribir los cores.
Cada corrida escribe su propio shard (results/<run_id> y loss/<run_id>) — sin contencion;
Part F hace glob de todos los shards.

Uso:
  python -m barrido.driver --stage E0 --workers 4
  python -m barrido.driver --stage E1 --workers 16 --out analisis/resultados/barrido
  python -m barrido.driver --stage E2 --limit 4          # subconjunto para probar
  python -m barrido.driver --stage E1 --cvxpy            # activa techo CVXPY en A1/A2 (lento)

E5 (OOD) NO se corre aqui: es solo-evaluacion de los modelos ya entrenados (ver
barrido/ood_eval.py y Parte E).
"""
import argparse
import os
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # .../Red
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                                             # load_vertices busca el .mat aqui

import torch                                               # noqa: E402

from barrido.config import build_stage                     # noqa: E402
from barrido.run import run_one                             # noqa: E402

DEFAULT_OUT = ROOT / "analisis" / "resultados" / "barrido"


def _init_worker():
    torch.set_num_threads(1)                               # un hilo por corrida (paralelismo por corrida)
    torch.set_default_dtype(torch.float64)


def _run_cfg(cfg, out_root, cvxpy):
    if cvxpy:
        cfg = cfg.__class__(**{**cfg.__dict__, "cvxpy_ceiling": True})
    return run_one(cfg, out_root)


def main():
    ap = argparse.ArgumentParser(description="Barrido experimental LMI-Net por etapas.")
    ap.add_argument("--stage", required=True, help="E0 | E1 | E2 | E3 | E4")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None, help="usar solo las primeras N corridas")
    ap.add_argument("--cvxpy", action="store_true", help="activar techo CVXPY en A1/A2")
    args = ap.parse_args()

    out_root = Path(args.out) / args.stage
    configs = build_stage(args.stage)
    if args.limit:
        configs = configs[:args.limit]
    print(f"Etapa {args.stage}: {len(configs)} corridas -> {out_root}  (workers={args.workers})")

    worker = partial(_run_cfg, out_root=str(out_root), cvxpy=args.cvxpy)
    results = []
    if args.workers == 1:
        _init_worker()
        for i, cfg in enumerate(configs, 1):
            r = worker(cfg)
            results.append(r)
            print(f"[{i}/{len(configs)}] {r['status']:5}  {r['run_id']}")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")                      # Windows-safe
        with ctx.Pool(args.workers, initializer=_init_worker) as pool:
            for i, r in enumerate(pool.imap_unordered(worker, configs), 1):
                results.append(r)
                print(f"[{i}/{len(configs)}] {r['status']:5}  {r['run_id']}")

    ok = sum(1 for r in results if r["status"] == "ok")
    err = [r for r in results if r["status"] != "ok"]
    print(f"\nListo: {ok}/{len(results)} ok, {len(err)} con error.")
    for r in err[:20]:
        print(f"  ERROR {r['run_id']}: {r.get('error')}")


if __name__ == "__main__":
    main()
