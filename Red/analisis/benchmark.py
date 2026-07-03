"""
benchmark.py
============
API de benchmark reutilizable para estabilizar sistemas politopicos con las
arquitecturas LMI-Net. Pensada para el notebook: eliges arquitectura e
hiperparametros, corre entrenamiento+evaluacion y devuelve METRICAS comparables
(tiempo, memoria pico, % estabilizado, loss, tiempo de inferencia/sistema) y
figuras automaticas.

Tambien permite meter TUS PROPIOS sistemas a mano (nominal o por vertices) y,
p. ej., desplazar los polos hacia un alpha para ver como responde el algoritmo.

Uso rapido:
    from analisis import benchmark as bm
    res = bm.run_experiment(arch="actuadores", n=3, N_list=[2,3,4,5], m=1,
                            dr_train=30, dr_eval=1000, epochs=15, implicit=False)
    bm.plot_experiment(res)                      # loss + estabilizacion + metricas
    # comparar configuraciones/arquitecturas:
    bm.compare([res_a, res_b], labels=["unroll", "implicit"])
"""
import time, threading
from pathlib import Path

import numpy as np, torch, pandas as pd
import matplotlib.pyplot as plt
import psutil

from entrenamiento.training import (load_vertices, split_items, make_batches, control_loss,
                                    paper_loss, get_loss_fn, get_loss_direction,
                                    register_loss, LOSS_REGISTRY)
from red.vanilla import LMINetVanilla
from red.vertices import LMINetVertices
from red.actuators import LMINetActuators

torch.set_default_dtype(torch.float64)

# arquitecturas seleccionables por nombre
ARCHS = {"vanilla": LMINetVanilla, "vertices": LMINetVertices, "actuadores": LMINetActuators}


# ============================ utilidades ==================================
def build_model(arch, n, m=1, N=2, dr_iters=30, alpha=0.01,
                backprop="unrolling", implicit=None, **kw):
    """Construye una arquitectura x metodo de backprop.
    arch:     'vanilla' (m,N fijos) | 'vertices' (inv. N) | 'actuadores' (inv. N y m).
    backprop: 'unrolling' | 'implicit'  (implicit=True/False es un alias legado)."""
    if implicit is not None:
        backprop = "implicit" if implicit else "unrolling"
    if arch == "vanilla":
        model = LMINetVanilla(n=n, m=m, N=N, dr_iters=dr_iters, alpha=alpha, backprop=backprop, **kw)
    elif arch == "vertices":
        model = LMINetVertices(n=n, m=m, dr_iters=dr_iters, alpha=alpha, backprop=backprop, **kw)
    elif arch == "actuadores":
        model = LMINetActuators(n=n, dr_iters=dr_iters, alpha=alpha, backprop=backprop, **kw)
    else:
        raise ValueError(f"arch desconocida: {arch}. Usa {list(ARCHS)}")
    return model.double()


def _forward(model, A, B):
    """Firma uniforme: todas las arquitecturas (incl. vanilla) reciben (A, B)."""
    return model(A, B)


class MemTracker:
    """Mide memoria pico (RSS) del proceso durante un bloque, muestreando en un hilo."""
    def __init__(self, dt=0.005):
        self.dt = dt; self.peak = 0; self.start = 0; self._run = False

    def __enter__(self):
        self.start = psutil.Process().memory_info().rss
        self.peak = self.start; self._run = True
        self._t = threading.Thread(target=self._sample, daemon=True); self._t.start()
        return self

    def _sample(self):
        p = psutil.Process()
        while self._run:
            self.peak = max(self.peak, p.memory_info().rss)
            time.sleep(self.dt)

    def __exit__(self, *a):
        self._run = False; self._t.join()
        self.peak_mb = self.peak / 1e6
        self.delta_mb = (self.peak - self.start) / 1e6      # memoria extra del bloque


# ============================ train / eval ================================
def _loss_for(model, arch, loss=None):
    """Resuelve la pérdida a usar: si `loss` se especifica (nombre del registro
    de training.LOSS_REGISTRY o un callable), esa gana. Si no, el default por
    arquitectura: vanilla usa 'paper' (receta fiel del apéndice); el resto 'control'."""
    if loss is not None:
        return get_loss_fn(loss, model)
    return get_loss_fn("paper" if arch == "vanilla" else "control", model)


def _train_loop(model, buckets, epochs, batch, lr, seed, loss_fn=None, log_every=0, tag=""):
    """Entrena sobre buckets {clave: [items (A,B)]}, cada bucket uniforme en (N,m).
    loss_fn(Q,Y,A,B); por defecto control_loss. Si log_every>0, imprime progreso
    (loss + tiempo/época) cada log_every épocas. Devuelve historial de loss por época."""
    if loss_fn is None:
        loss_fn = control_loss
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    model.train(); hist = []
    for ep in range(epochs):
        t_ep = time.perf_counter()
        batches = []
        for items in buckets.values():
            batches += make_batches(items, batch, shuffle=True, rng=rng)
        rng.shuffle(batches)
        run, nb = 0.0, 0
        for A, B in batches:
            opt.zero_grad(); Q, Y = _forward(model, A, B); loss = loss_fn(Q, Y, A, B)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); run += loss.item(); nb += 1
        hist.append(run / max(nb, 1))
        if log_every and (ep == 0 or (ep + 1) % log_every == 0):
            print(f"    {tag}época {ep + 1:>4}/{epochs}  loss={hist[-1]:.4f}  "
                  f"({time.perf_counter() - t_ep:.1f}s/época)", flush=True)
    return hist


@torch.no_grad()
def _eval_stab(model, items, dr_eval, alpha, batch):
    """% estabilizado, % con decay>=alpha, peor autovalor medio y tiempo de
    inferencia por sistema. La inferencia usa SIEMPRE el forward DR desenrollado
    (aunque el modelo se haya entrenado con implicita) para medir tiempo real."""
    old_it, old_impl = model.dr_iters, model.use_implicit
    model.dr_iters = dr_eval; model.use_implicit = False; model.eval()
    worst, t, ncount = [], 0.0, 0
    for A, B in make_batches(items, batch, shuffle=False):
        t0 = time.perf_counter(); Q, Y = _forward(model, A, B); t += time.perf_counter() - t0
        K = torch.bmm(Y, torch.linalg.inv(Q))
        for k in range(A.shape[0]):
            we = max(torch.linalg.eigvals(A[k, i] + B[k, i] @ K[k]).real.max().item()
                     for i in range(A.shape[1]))
            worst.append(we); ncount += 1
    model.dr_iters, model.use_implicit = old_it, old_impl
    w = np.array(worst)
    return dict(stable_pct=100 * (w < 0).mean(), decay_pct=100 * (w <= -alpha).mean(),
                worst_mean=float(w.mean()), infer_ms_per_sys=t / ncount * 1000, n=ncount)


# ------------------- guardado de fallos por combinacion -------------------
# Cada combinacion (arquitectura x backprop x n x m) tiene su PROPIO directorio:
#   resultados/benchmark/<arch>/<backprop>/n{n}_m{m}/{failures.csv, failures.npz, summary.csv}
# Asi se ve claramente que sistemas fallan segun la combinacion usada.
RESULTS_ROOT = Path(__file__).resolve().parent / "resultados" / "benchmark"


def _descriptors(A, B, N):
    Al = [A[i].numpy() for i in range(N)]; Bl = [B[i].numpy() for i in range(N)]
    Abar = np.mean(Al, 0)
    return dict(absc_ol=max(np.linalg.eigvals(Ai).real.max() for Ai in Al),
                normA=max(np.linalg.norm(Ai, 2) for Ai in Al),
                normB=max(np.linalg.norm(Bi, 2) for Bi in Bl),
                condA=max(np.linalg.cond(Ai) for Ai in Al),
                dispA=max(np.linalg.norm(Ai - Abar, 2) for Ai in Al))


def _combo_dir(arch, backprop, n, m_list):
    tag = "".join(str(x) for x in m_list)          # p.ej. [1,2] -> "12"
    d = RESULTS_ROOT / arch / backprop / f"n{n}_m{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


@torch.no_grad()
def _collect_failures(model, data, arch, backprop, n, dr_eval, alpha, batch):
    """Evalua por sistema el test y recolecta los NO estabilizados con su
    combinacion completa (arch, backprop, n, m, N) + descriptores + matrices A,B.
    `data` esta keyeado por (m, N); `model` es uno invariante o {N: model} (vanilla)."""
    def eval_one(mdl, items, mm, N):
        old_it, old_impl = mdl.dr_iters, mdl.use_implicit
        mdl.dr_iters = dr_eval; mdl.use_implicit = False; mdl.eval()
        rows, AB = [], []
        for A, B in make_batches(items, batch, shuffle=False):
            Q, Y = _forward(mdl, A, B); K = torch.bmm(Y, torch.linalg.inv(Q))
            for k in range(A.shape[0]):
                we = max(torch.linalg.eigvals(A[k, i] + B[k, i] @ K[k]).real.max().item()
                         for i in range(A.shape[1]))
                if we >= 0:                                     # NO estabilizado
                    rows.append(dict(arch=arch, backprop=backprop, n=n, m=mm, N=N,
                                     worst_eig=we, **_descriptors(A[k], B[k], A.shape[1])))
                    AB.append((A[k].numpy(), B[k].numpy()))
        mdl.dr_iters, mdl.use_implicit = old_it, old_impl
        return rows, AB

    rows, AB = [], []
    for (mm, N), (_, te) in data.items():
        mdl = model[N] if isinstance(model, dict) else model
        r, ab = eval_one(mdl, te, mm, N); rows += r; AB += ab
    return pd.DataFrame(rows), AB


def _save_failures(fdf, fAB, outdir, summary=None):
    fdf.to_csv(outdir / "failures.csv", index=False)
    np.savez_compressed(outdir / "failures.npz",
                        meta=(fdf[["n", "m", "N"]].to_numpy() if len(fdf) else np.empty((0, 3))),
                        **{f"A_{i}": ab[0] for i, ab in enumerate(fAB)},
                        **{f"B_{i}": ab[1] for i, ab in enumerate(fAB)})
    if summary is not None:
        summary.to_csv(outdir / "summary.csv", index=False)


def run_experiment(arch="actuadores", n=3, N_list=(2, 3, 4, 5), m=1, *,
                   dr_train=30, dr_eval=1000, epochs=15, batch=16, lr=1e-3,
                   alpha=0.01, limit=150, backprop="unrolling", implicit=None,
                   loss=None, loss_fn=None, seed=42, save_failures=False, log_every=0,
                   verbose=True, compare_cvxpy=True, cvxpy_max_systems=20):
    """Entrena UNA arquitectura y evalua. `m` puede ser un entero o una LISTA
    (p.ej. m=[1,2]): entonces el modelo se entrena sobre varios nº de actuadores a
    la vez (solo 'actuadores', que es invariante a n_u) y se evalua por (m, N).

    `loss`: nombre de una pérdida registrada en entrenamiento.training.LOSS_REGISTRY
    (hoy: 'control', 'paper'; añade la tuya con training.register_loss(nombre, factory)
    sin tocar este archivo). Si no se especifica, usa el default por arquitectura
    ('paper' para vanilla, 'control' para el resto). `loss_fn` (callable, legado)
    tiene prioridad sobre `loss` si ambos se pasan.

    Si compare_cvxpy (default True), cada fila del df tambien trae `cvxpy_pct`:
    % de sistemas de test con LMI factible segun CVXPY (verdad de terreno), para
    comparar contra el `stable_pct` logrado por la red.

    Devuelve dict con: model, df (una fila por (m, N)), loss_hist, t_train_s,
    mem_train_mb, stable_pct (media), infer_ms_per_sys, cvxpy_pct (media, si
    compare_cvxpy) y (si save_failures) el directorio de fallos de esta combinacion.
    """
    if implicit is not None:                       # alias legado
        backprop = "implicit" if implicit else "unrolling"
    m_list = [m] if isinstance(m, (int, np.integer)) else list(m)
    if len(m_list) > 1 and arch != "actuadores":
        raise ValueError("Solo arch='actuadores' admite varios m a la vez "
                         "(es la unica invariante al nº de actuadores).")
    N_list = list(N_list)
    data = {(mm, N): split_items(load_vertices(n, mm, N, limit=limit), seed=seed)
            for mm in m_list for N in N_list}
    loss_name = "custom" if loss_fn else (loss or ("paper" if arch == "vanilla" else "control"))
    loss_direction = get_loss_direction(loss_name)
    loss_vs_cvxpy_frames = []

    if arch == "vanilla":
        # vanilla NO es invariante a N ni a m: un modelo por N (m es unico).
        mm = m_list[0]
        models, df_rows, hist, t_train, mem = {}, [], [], 0.0, 0.0
        for N in N_list:
            mdl = build_model(arch, n, mm, N=N, dr_iters=dr_train, alpha=alpha, backprop=backprop)
            loss_used = loss_fn or _loss_for(mdl, arch, loss)
            with MemTracker() as mt:
                t0 = time.perf_counter()
                hist = _train_loop(mdl, {(mm, N): data[(mm, N)][0]}, epochs, batch, lr, seed,
                                   loss_fn=loss_used, log_every=log_every, tag=f"[N={N}] ")
                t_train += time.perf_counter() - t0
            mem = max(mem, mt.delta_mb)
            row = dict(m=mm, N=N, **_eval_stab(mdl, data[(mm, N)][1], dr_eval, alpha, batch))
            if compare_cvxpy:
                (row["cvxpy_pct"], row["cvxpy_ms_per_sys"]), per_sys = _cvxpy_loss_stats(
                    mdl, data[(mm, N)][1], loss_used, N, mm, cvxpy_max_systems)
                loss_vs_cvxpy_frames.append(per_sys)
            df_rows.append(row)
            models[N] = mdl
        model = models
        df = pd.DataFrame(df_rows)
    else:
        model = build_model(arch, n, m_list[0], dr_iters=dr_train, alpha=alpha, backprop=backprop)
        loss_used = loss_fn or _loss_for(model, arch, loss)
        with MemTracker() as mt:
            t0 = time.perf_counter()
            hist = _train_loop(model, {k: data[k][0] for k in data}, epochs, batch, lr, seed,
                               loss_fn=loss_used, log_every=log_every)
            t_train = time.perf_counter() - t0
        mem = mt.delta_mb
        rows = []
        for mm in m_list:
            for N in N_list:
                row = dict(m=mm, N=N, **_eval_stab(model, data[(mm, N)][1], dr_eval, alpha, batch))
                if compare_cvxpy:
                    (row["cvxpy_pct"], row["cvxpy_ms_per_sys"]), per_sys = _cvxpy_loss_stats(
                        model, data[(mm, N)][1], loss_used, N, mm, cvxpy_max_systems)
                    loss_vs_cvxpy_frames.append(per_sys)
                rows.append(row)
        df = pd.DataFrame(rows)

    loss_vs_cvxpy = pd.concat(loss_vs_cvxpy_frames, ignore_index=True) if loss_vs_cvxpy_frames else None
    pct_red_mejor = None
    if loss_vs_cvxpy is not None and len(loss_vs_cvxpy):
        loss_vs_cvxpy["red_mejor"] = (loss_vs_cvxpy.loss_red < loss_vs_cvxpy.loss_cvxpy
                                      if loss_direction == "min" else
                                      loss_vs_cvxpy.loss_red > loss_vs_cvxpy.loss_cvxpy)
        pct_red_mejor = float(loss_vs_cvxpy.red_mejor.mean() * 100)

    res = dict(arch=arch, n=n, m=m, m_list=m_list, N_list=N_list, model=model, df=df,
               loss_hist=hist, t_train_s=t_train, mem_train_mb=mem,
               stable_pct=float(df.stable_pct.mean()),
               infer_ms_per_sys=float(df.infer_ms_per_sys.mean()),
               cvxpy_pct=(float(df.cvxpy_pct.mean()) if "cvxpy_pct" in df else None),
               cvxpy_ms_per_sys=(float(df.cvxpy_ms_per_sys.mean()) if "cvxpy_ms_per_sys" in df else None),
               loss_vs_cvxpy=loss_vs_cvxpy, pct_red_mejor_loss=pct_red_mejor,
               config=dict(dr_train=dr_train, dr_eval=dr_eval, epochs=epochs, batch=batch,
                           lr=lr, alpha=alpha, limit=limit, backprop=backprop, seed=seed,
                           loss=loss_name, loss_direction=loss_direction))
    if verbose:
        cvxpy_msg = (f"  cvxpy_factible={res['cvxpy_pct']:.1f}%  "
                     f"cvxpy_infer={res['cvxpy_ms_per_sys']:.1f}ms/sys"
                     if res["cvxpy_pct"] is not None else "")
        mejor_msg = (f"  red_mejor_que_cvxpy({loss_direction} {loss_name})={pct_red_mejor:.1f}%"
                     if pct_red_mejor is not None else "")
        print(f"[{arch}/{backprop}] n={n} m={m_list}  estabiliza={res['stable_pct']:.1f}%{cvxpy_msg}"
              f"{mejor_msg}  "
              f"train={t_train:.1f}s  mem_pico=+{mem:.0f}MB  "
              f"inferencia={res['infer_ms_per_sys']:.2f}ms/sys  loss_final={hist[-1]:.4f}")

    if save_failures:
        fdf, fAB = _collect_failures(model, data, arch, backprop, n, dr_eval, alpha, batch)
        outdir = _combo_dir(arch, backprop, n, m_list)
        _save_failures(fdf, fAB, outdir, summary=df)
        res["failures_dir"] = str(outdir); res["n_failures"] = len(fdf)
        if verbose:
            print(f"  -> {len(fdf)} sistemas NO estabilizados guardados en {outdir}")
    return res


# ============================ grid search ==================================
def grid_search(param_grid, verbose=False, **common):
    """Barre TODAS las combinaciones de `param_grid` (dict {parametro: [valores]},
    p.ej. {"loss": ["control","paper"], "backprop": ["unrolling","implicit"]}) sobre
    run_experiment, con `common` como el resto de kwargs fijos (arch, n, N_list, m,
    epochs, limit, ...). Cualquier parametro de run_experiment es barrible, incluida
    'loss' (nombres del registro en entrenamiento.training.LOSS_REGISTRY).

    Devuelve (tabla, resultados):
      - tabla: un DataFrame con una fila por combinacion + metricas resumen
        (stable_pct, t_train_s, mem_train_mb, infer_ms_per_sys, loss_final, cvxpy_pct,
        y si compare_cvxpy: pct_red_mejor_loss — % de sistemas donde la red logra
        mejor valor de `loss` que el certificado FACTIBLE de CVXPY, ver
        plot_loss_vs_cvxpy).
      - resultados: lista de dicts run_experiment completos, mismo orden que `tabla`.
    """
    import itertools
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    rows, results = [], []
    for i, combo in enumerate(combos):
        kwargs = dict(zip(keys, combo))
        tag = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"\n=== grid {i + 1}/{len(combos)}: {tag} ===")
        r = run_experiment(**{**common, **kwargs}, verbose=verbose)
        row = {**kwargs, "stable_pct": r["stable_pct"], "t_train_s": r["t_train_s"],
              "mem_train_mb": r["mem_train_mb"], "infer_ms_per_sys": r["infer_ms_per_sys"],
              "loss_final": r["loss_hist"][-1], "loss_usada": r["config"]["loss"]}
        if r["cvxpy_pct"] is not None:
            row["cvxpy_pct"] = r["cvxpy_pct"]
        if r["pct_red_mejor_loss"] is not None:
            row["pct_red_mejor_loss"] = r["pct_red_mejor_loss"]
        print(f"  -> estabiliza={r['stable_pct']:.1f}%  train={r['t_train_s']:.1f}s  "
              f"loss_final={row['loss_final']:.4f}")
        rows.append(row); results.append(r)
    return pd.DataFrame(rows), results


def plot_grid_search(tabla, x, y="stable_pct", hue=None, title=None):
    """Barras de la métrica `y` vs el parámetro `x` del grid (de grid_search),
    agrupadas por `hue` si se especifica (p.ej. x='loss', hue='backprop')."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if hue and hue in tabla.columns:
        hues = sorted(tabla[hue].astype(str).unique())
        xs = sorted(tabla[x].astype(str).unique())
        xpos = np.arange(len(xs)); w = 0.8 / len(hues)
        t = tabla.copy(); t[x] = t[x].astype(str); t[hue] = t[hue].astype(str)
        for i, hv in enumerate(hues):
            sub = t[t[hue] == hv].set_index(x).reindex(xs)
            b = ax.bar(xpos + (i - (len(hues) - 1) / 2) * w, sub[y], w, label=f"{hue}={hv}", alpha=.85)
            ax.bar_label(b, fmt="%.1f", fontsize=7)
        ax.set_xticks(xpos); ax.set_xticklabels(xs); ax.legend(fontsize=8)
    else:
        xs = tabla[x].astype(str)
        b = ax.bar(xs, tabla[y], alpha=.85)
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set(xlabel=x, ylabel=y, title=title or f"Grid search: {y} vs {x}")
    fig.tight_layout(); return fig


# ==================== sistemas a mano (para el profesor) =================
def system_to_polytope(A, B):
    """Un sistema NOMINAL (no politopo) -> formato del modelo con N=1.
    A: (n,n) lista/np/tensor.  B: (n,m).  Devuelve A_poly (1,n,n), B_poly (1,n,m)."""
    A = torch.as_tensor(np.atleast_2d(np.asarray(A, float))).double()
    B = torch.as_tensor(np.atleast_2d(np.asarray(B, float))).double()
    if B.shape[0] != A.shape[0]:            # por si pasan B como fila (m,n) o (n,)
        B = B.reshape(A.shape[0], -1)
    return A.unsqueeze(0), B.unsqueeze(0)


def polytope_from_vertices(A_list, B_list):
    """Un POLITOPO por sus vertices. A_list=[A_1,...,A_N], B_list=[B_1,...,B_N].
    Devuelve A_poly (N,n,n), B_poly (N,n,m)."""
    A = torch.stack([torch.as_tensor(np.atleast_2d(np.asarray(a, float))) for a in A_list]).double()
    B = torch.stack([torch.as_tensor(np.asarray(b, float)).reshape(A.shape[1], -1) for b in B_list]).double()
    return A, B


def shift_poles(A_poly, shift):
    """Desplaza TODOS los polos (parte real de los autovalores de A) sumando
    `shift`·I a cada vertice:  A -> A + shift·I.  shift>0 hace el sistema mas
    inestable; shift<0 lo hace mas estable. Util para barrer la 'dificultad'."""
    n = A_poly.shape[-1]
    return A_poly + shift * torch.eye(n, dtype=A_poly.dtype)


def normalize_system(A_poly, B_poly):
    """Misma normalizacion fisica que el data_loader: divide por el mayor valor
    absoluto de A y B (los pone en [-1,1]). El modelo fue entrenado asi."""
    gamma = torch.max(A_poly.abs().max(), B_poly.abs().max()).clamp(min=1e-12)
    return A_poly / gamma, B_poly / gamma


_ARCH_NAME = {"LMINetVanilla": "vanilla", "LMINetVertices": "vertices", "LMINetActuators": "actuadores"}


@torch.no_grad()
def benchmark_systems(model, systems, dr_eval=1000, alpha=0.01,
                      normalize=True, labels=None, compare_cvxpy=True, loss=None,
                      save_failures=False):
    """Evalua el modelo sobre TUS sistemas. `systems` = lista de politopos
    (A_poly, B_poly) construidos con system_to_polytope / polytope_from_vertices.

    Por sistema devuelve: estabilizado, peor autovalor de lazo cerrado, decay
    logrado, tiempo de inferencia (`infer_ms`). Con compare_cvxpy=True (default),
    tambien si la LMI es factible por CVXPY (verdad de terreno: existe certificado
    con margen alpha, columna `lmi_factible_cvxpy`) y su tiempo de solve
    (`cvxpy_ms`) — para comparar directamente contra `infer_ms`.

    Si ademas se pasa `loss` (nombre registrado en training.LOSS_REGISTRY, p.ej.
    "control"/"paper", o un callable), se añaden `loss_red`/`loss_cvxpy`: el valor
    de esa pérdida evaluado en el (Q,Y) de la red y en el de CVXPY. Importante:
    CVXPY solo resuelve FACTIBILIDAD (el certificado mas cercano a y=0), no
    optimiza ningun criterio de desempeño -- por eso puede no minimizar/maximizar
    `loss` aunque sea una solucion valida; esta comparacion muestra si la red
    encuentra soluciones mejores (ver plot_loss_vs_cvxpy).

    Si save_failures, guarda los NO estabilizados en el directorio de la combinacion
    resultados/benchmark/<arch>/<backprop>/manual/.
    """
    if isinstance(model, dict):
        raise ValueError("benchmark_systems necesita un modelo invariante "
                         "(vertices/actuadores), no el dict per-N de vanilla.")
    loss_fn = get_loss_fn(loss, model) if loss is not None else None
    old_it, old_impl = model.dr_iters, model.use_implicit
    model.dr_iters = dr_eval; model.use_implicit = False; model.eval()
    rows, fail_AB = [], []
    for idx, (A, B) in enumerate(systems):
        A = torch.as_tensor(A).double(); B = torch.as_tensor(B).double()
        if normalize:
            A, B = normalize_system(A, B)
        Ab, Bb = A.unsqueeze(0), B.unsqueeze(0)                 # batch 1
        t0 = time.perf_counter(); Q, Y = _forward(model, Ab, Bb); dt = time.perf_counter() - t0
        K = (Y[0] @ torch.linalg.inv(Q[0]))
        we = max(torch.linalg.eigvals(A[i] + B[i] @ K).real.max().item() for i in range(A.shape[0]))
        row = dict(sistema=(labels[idx] if labels else idx), N=A.shape[0], n=A.shape[1],
                   m=B.shape[-1], estabilizado=bool(we < 0), peor_autovalor_cl=we,
                   decay_logrado=-we, infer_ms=dt * 1000)
        if compare_cvxpy:
            feas, t_cvx, Qc, Yc = _cvxpy_solve(model, A, B)
            row["lmi_factible_cvxpy"] = feas
            row["cvxpy_ms"] = t_cvx * 1000 if t_cvx is not None else float("nan")
            if loss_fn is not None and feas:
                row["loss_red"] = float(loss_fn(Q, Y, Ab, Bb))
                row["loss_cvxpy"] = float(loss_fn(Qc, Yc, Ab, Bb))
        rows.append(row)
        if we >= 0:
            fail_AB.append((A.numpy(), B.numpy()))
    model.dr_iters, model.use_implicit = old_it, old_impl
    df = pd.DataFrame(rows)

    if save_failures:
        arch = _ARCH_NAME.get(type(model).__name__, type(model).__name__)
        outdir = RESULTS_ROOT / arch / model.backprop / "manual"; outdir.mkdir(parents=True, exist_ok=True)
        df[~df.estabilizado].to_csv(outdir / "failures.csv", index=False)
        np.savez_compressed(outdir / "failures.npz",
                            **{f"A_{i}": ab[0] for i, ab in enumerate(fail_AB)},
                            **{f"B_{i}": ab[1] for i, ab in enumerate(fail_AB)})
        print(f"  -> {len(fail_AB)} sistemas NO estabilizados guardados en {outdir}")
    return df


def _cvxpy_solve(model, A, B):
    """Resuelve la proyeccion LMI por CVXPY para UN sistema: busca el certificado
    (Q,Y) FACTIBLE mas cercano a y=0 (verdad de terreno de factibilidad, NO
    optimiza ningun criterio de desempeño -- por eso se compara aparte contra la
    perdida de la red, ver _cvxpy_loss_stats).
    A: (N,n,n), B: (N,n,m) de UN sistema (sin dim de batch).
    Devuelve (factible: bool|None, t_solve_s: float|None, Q: (1,n,n)|None, Y: (1,m,n)|None)."""
    from analisis.validate_projection import cvxpy_projection
    old_N = model.N
    model.N = A.shape[0]           # cvxpy_projection lee model.N; debe ser el N real del sistema
    try:
        y0 = np.zeros(model.dim_y)
        y_star, t_solve = cvxpy_projection(model, y0, A.numpy(), B.numpy())
        y_t = torch.from_numpy(y_star).to(A.dtype).unsqueeze(0)
        Q, Y = model._y_to_matrices(y_t)
        return True, t_solve, Q, Y
    except Exception:
        return None, None, None, None
    finally:
        model.N = old_N


def _cvxpy_loss_stats(model, items, loss_fn, N, m, max_systems=20):
    """% factible y tiempo medio de solve (ms/sistema) segun CVXPY (verdad de
    terreno + costo de referencia), sobre a lo sumo `max_systems` de `items`
    (una SDP por sistema es lento; con esto el costo queda acotado aunque
    `limit`/N_list crezcan).
    # ponytail: muestra los primeros max_systems en vez de todos; sube max_systems
    # si necesitas el % / tiempo exacto sobre el set de test completo.

    ADEMAS compara, sistema por sistema, `loss_fn` evaluada en el (Q,Y) de la RED
    contra la evaluada en el (Q,Y) de CVXPY. CVXPY solo busca un certificado
    FACTIBLE (el mas cercano a y=0), no optimiza `loss_fn` -- por eso puede no ser
    el mejor segun el criterio que de verdad importa (p.ej. volumen del elipsoide),
    aunque sea una solucion valida. Esta comparacion muestra si la red encuentra
    soluciones mejores que "una cualquiera factible".

    Devuelve ((pct_factible, ms_per_sys), per_sys: DataFrame con columnas
    N, m, loss_red, loss_cvxpy)."""
    sample = items[:max_systems]
    feas_list, times, rows = [], [], []
    for A, B in sample:
        feas, t_solve, Qc, Yc = _cvxpy_solve(model, A, B)
        if feas is None:
            continue
        feas_list.append(1); times.append(t_solve)
        Ab, Bb = A.unsqueeze(0), B.unsqueeze(0)
        with torch.no_grad():
            Qr, Yr = _forward(model, Ab, Bb)
        loss_red = float(loss_fn(Qr, Yr, Ab, Bb))
        loss_cvxpy = float(loss_fn(Qc, Yc, Ab, Bb))
        rows.append(dict(N=N, m=m, loss_red=loss_red, loss_cvxpy=loss_cvxpy))
    pct = 100 * len(feas_list) / len(sample) if sample else float("nan")
    ms_per_sys = 1000 * np.mean(times) if times else float("nan")
    return (pct, ms_per_sys), pd.DataFrame(rows)


# ============================ figuras ====================================
def plot_experiment(res):
    """Loss por epoca + % estabilizado por N + tiempo de inferencia (red vs CVXPY) por N."""
    df = res["df"]
    has_cvxpy_ms = "cvxpy_ms_per_sys" in df
    n_panels = 3 if has_cvxpy_ms else 2
    fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 3.4))
    axs[0].plot(range(1, len(res["loss_hist"]) + 1), res["loss_hist"], marker=".")
    axs[0].set(xlabel="epoca", ylabel="loss", title=f"Entrenamiento [{res['arch']}]")
    ms = sorted(df.m.unique()); Ns = sorted(df.N.unique())
    x = np.arange(len(Ns)); w = 0.8 / len(ms)
    for i, mm in enumerate(ms):
        sub = df[df.m == mm].set_index("N").reindex(Ns)
        b = axs[1].bar(x + (i - (len(ms) - 1) / 2) * w, sub.stable_pct, w,
                       label=f"m={mm}", alpha=.85)
        axs[1].bar_label(b, fmt="%.0f", fontsize=7)
    axs[1].set_xticks(x); axs[1].set_xticklabels(Ns)
    axs[1].set(xlabel="N (vertices)", ylabel="% estabilizados", ylim=(0, 100),
               title="Estabilizacion por N" + (" y m" if len(ms) > 1 else ""))
    if len(ms) > 1:
        axs[1].legend(fontsize=8)
    if has_cvxpy_ms:
        # tiempo de inferencia red vs CVXPY, agregado por N (media sobre m)
        sub = df.groupby("N")[["infer_ms_per_sys", "cvxpy_ms_per_sys"]].mean().reindex(Ns)
        w2 = 0.35
        b1 = axs[2].bar(x - w2 / 2, sub.infer_ms_per_sys, w2, label="red", color="#1e8449", alpha=.85)
        b2 = axs[2].bar(x + w2 / 2, sub.cvxpy_ms_per_sys, w2, label="CVXPY", color="#a93226", alpha=.85)
        axs[2].bar_label(b1, fmt="%.1f", fontsize=7); axs[2].bar_label(b2, fmt="%.0f", fontsize=7)
        axs[2].set_yscale("log")
        axs[2].set_xticks(x); axs[2].set_xticklabels(Ns)
        axs[2].set(xlabel="N (vertices)", ylabel="ms/sistema (log)",
                   title="Inferencia: red vs CVXPY")
        axs[2].legend(fontsize=8)
    c = res["config"]
    cvxpy_txt = (f" | cvxpy {res['cvxpy_pct']:.1f}% ({res['cvxpy_ms_per_sys']:.0f}ms/sys)"
                if res.get("cvxpy_pct") is not None else "")
    fig.suptitle(f"{res['arch']} / {c['backprop']} | n={res['n']} m={res['m']} | "
                 f"estabiliza {res['stable_pct']:.1f}%{cvxpy_txt} | train {res['t_train_s']:.0f}s | "
                 f"mem +{res['mem_train_mb']:.0f}MB | infer {res['infer_ms_per_sys']:.2f}ms/sys | "
                 f"DR_tr={c['dr_train']} DR_ev={c['dr_eval']}",
                 fontsize=9, y=1.03)
    fig.tight_layout(); return fig


def compare(results, labels=None,
           metrics=("stable_pct", "cvxpy_pct", "t_train_s", "mem_train_mb")):
    """Compara varias corridas (arquitecturas / configs) en barras. Incluye
    `cvxpy_pct` (verdad de terreno) junto a `stable_pct`, y SIEMPRE un panel de
    tiempo de inferencia red-vs-CVXPY (log-scale, mismo panel para comparar
    directamente) si `infer_ms_per_sys`/`cvxpy_ms_per_sys` estan presentes.
    Cualquier metrica ausente en algun resultado (p.ej. corrida con
    compare_cvxpy=False) se salta sola."""
    # infer_ms_per_sys / cvxpy_ms_per_sys SIEMPRE van en el panel combinado de abajo,
    # nunca como barra generica (aunque vengan incluidas en `metrics`).
    metrics = [mk for mk in metrics if mk not in ("infer_ms_per_sys", "cvxpy_ms_per_sys")
              and all(r.get(mk) is not None for r in results)]
    labels = labels or [f"{r['arch']}#{i}" for i, r in enumerate(results)]
    titles = {"stable_pct": "% estabilizados", "cvxpy_pct": "% factible LMI (CVXPY)",
              "t_train_s": "tiempo train [s]", "mem_train_mb": "mem pico [MB]"}
    has_infer = all(r.get("infer_ms_per_sys") is not None for r in results)
    has_cvxpy_ms = all(r.get("cvxpy_ms_per_sys") is not None for r in results)
    n_panels = len(metrics) + (1 if has_infer else 0)
    fig, axs = plt.subplots(1, n_panels, figsize=(3.2 * n_panels, 3.2))
    axs = np.atleast_1d(axs)
    for ax, mkey in zip(axs, metrics):
        vals = [r[mkey] for r in results]
        b = ax.bar(labels, vals, color="#1e8449", alpha=.8)
        ax.bar_label(b, fmt="%.1f", fontsize=8); ax.set_title(titles.get(mkey, mkey))
        ax.tick_params(axis="x", rotation=20)
    if has_infer:
        ax = axs[-1]
        x = np.arange(len(labels)); w = 0.35 if has_cvxpy_ms else 0.7
        b1 = ax.bar(x - (w / 2 if has_cvxpy_ms else 0), [r["infer_ms_per_sys"] for r in results],
                    w, label="red", color="#1e8449", alpha=.85)
        ax.bar_label(b1, fmt="%.1f", fontsize=7)
        if has_cvxpy_ms:
            b2 = ax.bar(x + w / 2, [r["cvxpy_ms_per_sys"] for r in results], w,
                       label="CVXPY", color="#a93226", alpha=.85)
            ax.bar_label(b2, fmt="%.0f", fontsize=7)
            ax.set_yscale("log")
            ax.legend(fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
        ax.set_title("inferencia [ms/sys]" + (" (log, red vs CVXPY)" if has_cvxpy_ms else ""))
    fig.tight_layout(); return fig


def compare_grid(archs=("vertices", "actuadores"), backprops=("unrolling", "implicit"),
                 metrics=("stable_pct", "cvxpy_pct", "t_train_s", "mem_train_mb",
                          "infer_ms_per_sys", "cvxpy_ms_per_sys"),
                 **common):
    """Corre TODAS las combinaciones arquitectura x backprop y las compara.
    `common` son los hiperparametros compartidos (n, N_list, m, dr_train, epochs, ...).
    Devuelve (tabla_resumen, results, figura)."""
    results, labels = [], []
    for arch in archs:
        for bp in backprops:
            results.append(run_experiment(arch=arch, backprop=bp, **common))
            labels.append(f"{arch}/{bp[:5]}")
    tabla = pd.DataFrame([{"arch": r["arch"], "backprop": r["config"]["backprop"],
                           **{k: r[k] for k in metrics}} for r in results])
    return tabla, results, compare(results, labels=labels, metrics=metrics)


def plot_pole_shift(df, x="shift"):
    """Para el barrido de pole-shift: decay logrado/estabilizacion vs shift, y
    (si esta la columna cvxpy_ms) tiempo de inferencia red vs CVXPY vs shift."""
    has_cvxpy_ms = "cvxpy_ms" in df
    fig, axs = plt.subplots(1, 2 if has_cvxpy_ms else 1,
                            figsize=(12.4 if has_cvxpy_ms else 6.4, 3.6))
    ax = axs[0] if has_cvxpy_ms else axs
    ax.plot(df[x], df["decay_logrado"], marker="o", color="#2471a3", label="decay logrado")
    ax.axhline(0, ls="--", color="k", alpha=.5)
    ax.fill_between(df[x], 0, df["decay_logrado"].max(), where=df["estabilizado"],
                    color="#2471a3", alpha=.08)
    ax.set(xlabel="shift aplicado a los polos (A -> A + shift·I)",
           ylabel="decay logrado  -max Re λ(A+BK)", title="Respuesta al desplazamiento de polos")
    ax.legend()
    if has_cvxpy_ms:
        axs[1].plot(df[x], df["infer_ms"], marker="o", color="#1e8449", label="red")
        axs[1].plot(df[x], df["cvxpy_ms"], marker="o", color="#a93226", label="CVXPY")
        axs[1].set_yscale("log")
        axs[1].set(xlabel="shift aplicado a los polos", ylabel="tiempo [ms] (log)",
                   title="Inferencia: red vs CVXPY")
        axs[1].legend()
    fig.tight_layout(); return fig


def plot_loss_vs_cvxpy(res, title=None):
    """Compara, SISTEMA POR SISTEMA, el valor de la pérdida de entrenamiento
    logrado por la red frente al logrado por el certificado (Q,Y) de CVXPY.

    Por qué esta comparación importa: CVXPY (en _cvxpy_solve) solo resuelve un
    problema de FACTIBILIDAD -- el certificado más cercano a y=0 que cumple la
    LMI -- no optimiza ningún criterio de desempeño. Es "verdad de terreno" de
    que existe solución, pero no de que sea buena. La red, en cambio, SÍ está
    entrenada para minimizar/maximizar `loss`. Este gráfico muestra si la red
    encuentra soluciones mejores que "una cualquiera factible" bajo ese criterio
    (p.ej. menor volumen del elipsoide invariante).

    Requiere haber corrido run_experiment con compare_cvxpy=True (default).
    Panel izquierdo: scatter loss_red vs loss_cvxpy con la diagonal y=x (empate).
    Panel derecho: histograma de la mejora (+ = la red es mejor), según la
    DIRECCIÓN de la pérdida (min/max, ver training.LOSS_REGISTRY).
    """
    df = res.get("loss_vs_cvxpy")
    if df is None or len(df) == 0:
        raise ValueError("No hay datos loss_vs_cvxpy: corre run_experiment con "
                         "compare_cvxpy=True (default) y al menos 1 sistema factible.")
    direction = res["config"]["loss_direction"]
    loss_name = res["config"]["loss"]
    pct = res["pct_red_mejor_loss"]

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))

    lo = min(df.loss_red.min(), df.loss_cvxpy.min())
    hi = max(df.loss_red.max(), df.loss_cvxpy.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    axs[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, alpha=.5, label="y = x (empate)")
    colors = np.where(df.red_mejor, "#1e8449", "#a93226")
    axs[0].scatter(df.loss_cvxpy, df.loss_red, c=colors, alpha=.7, s=24)
    lado = "mas ABAJO" if direction == "min" else "mas ARRIBA"
    axs[0].set(xlabel="pérdida con (Q,Y) de CVXPY", ylabel="pérdida con (Q,Y) de la red",
              title=f"Por sistema (red mejor = {lado} de la diagonal)")
    axs[0].legend(fontsize=8)

    delta = (df.loss_cvxpy - df.loss_red) if direction == "min" else (df.loss_red - df.loss_cvxpy)
    axs[1].hist(delta, bins=min(20, max(5, len(delta) // 2)), color="#2471a3", alpha=.85)
    axs[1].axvline(0, color="k", ls="--", lw=1)
    axs[1].set(xlabel="mejora de la red sobre CVXPY  (+ = red mejor)",
              ylabel="nº de sistemas", title=f"{pct:.1f}% de sistemas: la red es mejor")

    fig.suptitle(title or f"Pérdida ({loss_name}, {direction}) — red vs CVXPY  "
                          f"[{res['arch']}/{res['config']['backprop']}]", y=1.03)
    fig.tight_layout(); return fig
