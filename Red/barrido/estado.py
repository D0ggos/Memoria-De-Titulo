"""
estado.py  —  avance en vivo del barrido (se puede correr mientras corre el driver)
===================================================================================
Lee los shards ya escritos en disco (cada corrida escribe el suyo al terminar) y muestra:
  - progreso por etapa (completas / esperadas) y global, con ETA estimado por los mtimes
    de los shards meta;
  - un ranking PARCIAL de las mejores configs vistas hasta ahora (anillo A2);
  - las ultimas lineas del log de progreso.

No interfiere con el driver (solo lee). Uso:
  python -m barrido.estado
  python -m barrido.estado --watch 60      # refresca cada 60 s (Ctrl-C para salir)
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:                                                       # consola Windows (cp1252) -> UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from barrido.config import build_stage                    # noqa: E402
from barrido.reporte import (DEFAULT_BASE, ALL_STAGES, read_results,   # noqa: E402
                             leaderboard, count_meta, md_table, LB_COLS, LB_HEAD, _fmt)


def _fmt_dur(seconds):
    if seconds != seconds or seconds < 0:
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    d, h = divmod(h, 24)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def _meta_mtimes(base, stages):
    """(mas antiguo, mas nuevo) de todos los shards meta; sirve para estimar ritmo/ETA."""
    ts = []
    for st in stages:
        folder = Path(base) / st / "meta"
        if folder.exists():
            ts += [p.stat().st_mtime for p in
                   list(folder.glob("*.parquet")) + list(folder.glob("*.csv"))]
    return (min(ts), max(ts)) if ts else (None, None)


def snapshot(base, stages):
    done = count_meta(base, stages)
    expected = {st: len(build_stage(st)) for st in stages}
    tot_done = sum(done.values())
    tot_exp = sum(expected.values())

    print("=" * 64)
    print(f"AVANCE DEL BARRIDO   {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 64)
    hdr = f"{'etapa':6} {'completas':>10} {'esperadas':>10} {'%':>6}"
    print(hdr)
    for st in stages:
        e = expected[st]
        pct = f"{100*done[st]/e:.0f}%" if e else "—"
        print(f"{st:6} {done[st]:>10} {e:>10} {pct:>6}")
    gpct = f"{100*tot_done/tot_exp:.1f}%" if tot_exp else "—"
    print("-" * 34)
    print(f"{'TOTAL':6} {tot_done:>10} {tot_exp:>10} {gpct:>6}")

    # --- ritmo / ETA por mtimes de los meta ---
    t_old, t_new = _meta_mtimes(base, stages)
    if t_old and tot_done > 1:
        elapsed = t_new - t_old
        rate = elapsed / max(tot_done - 1, 1)             # s/corrida (ventana observada)
        eta = rate * (tot_exp - tot_done)
        since = time.time() - t_new
        print(f"\nRitmo observado: ~{_fmt_dur(rate)}/corrida  |  "
              f"trabajado: {_fmt_dur(elapsed)}  |  última corrida hace {_fmt_dur(since)}")
        print(f"ETA para completar ({tot_exp - tot_done} corridas restantes): "
              f"~{_fmt_dur(eta)}")

    # --- ranking parcial A2 ---
    df = read_results(base, stages)
    if not df.empty:
        lb, (ep, dre) = leaderboard(df, "A2")
        if not lb.empty:
            print(f"\nTop configs PARCIAL — A2 (época {ep}, dr_eval={dre}):")
            print(md_table(lb, LB_COLS, [h.replace("A? ", "A2 ") for h in LB_HEAD], max_rows=8))

    # --- cola del log ---
    log = Path(base) / "progreso.log"
    if log.exists():
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        print("Últimas líneas de progreso.log:")
        for ln in tail:
            print("  " + ln)


def main():
    ap = argparse.ArgumentParser(description="Avance en vivo del barrido.")
    ap.add_argument("--base", default=str(DEFAULT_BASE))
    ap.add_argument("--stages", default="ALL", help="ALL o lista 'E1,E4'")
    ap.add_argument("--watch", type=int, default=0, help="refrescar cada N segundos")
    args = ap.parse_args()
    stages = ALL_STAGES if args.stages.upper() == "ALL" else \
        [s.strip() for s in args.stages.split(",") if s.strip()]

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")            # limpia pantalla
                snapshot(args.base, stages)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n(fin del watch)")
    else:
        snapshot(args.base, stages)


if __name__ == "__main__":
    main()
