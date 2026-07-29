"""Calibracion de tiempos del barrido en CPU para estimar el total.
Mide, para la config base de E1 (actuadores, n_x=3, sz=150):
  - s/epoca de entrenamiento para cada dr_train (unrolling) y para implicit
  - costo de UNA pasada de evaluacion (A1+A2, ladder dr_eval completo a 8000)
No corre las 400 epocas: mide pocas y extrapola (el entrenamiento es lineal en epocas).
"""
import time, torch
from barrido.config import RunConfig, DR_TRAIN, BASE_ALPHA
from barrido.run import run_one

torch.set_num_threads(torch.get_num_threads())  # usa todos los hilos disponibles
OUT = "analisis/resultados/_calib"


def t_train_per_epoch(dr, backprop, epochs):
    """Corre `epochs` SIN evaluar (epoch_ckpts vacio) -> s/epoca de entrenamiento."""
    cfg = RunConfig(stage="CAL", loss="paper", arch="actuadores",
                    backprop=backprop, dr_train=dr, alpha=BASE_ALPHA, n_x=3,
                    train_size=150, seed=42, epochs=epochs, epoch_ckpts=(),
                    dr_eval=(100,))
    t0 = time.perf_counter()
    r = run_one(cfg, OUT, device="cpu")
    dt = time.perf_counter() - t0
    return r.get("t_train_s", float("nan")) / epochs, dt


def t_eval_one_ckpt(dr=5):
    """Corre 1 epoca con 1 checkpoint -> aisla el costo de UNA pasada de eval (A1+A2)."""
    cfg = RunConfig(stage="CAL", loss="paper", arch="actuadores",
                    backprop="unrolling", dr_train=dr, alpha=BASE_ALPHA, n_x=3,
                    train_size=150, seed=42, epochs=1, epoch_ckpts=(1,))
    t0 = time.perf_counter()
    r = run_one(cfg, OUT, device="cpu")
    wall = time.perf_counter() - t0
    t_train = r.get("t_train_s", 0.0)
    return wall - t_train, wall   # eval_por_checkpoint (ambos anillos), wall


print("=== CALIBRACION CPU (hilos=%d) ===" % torch.get_num_threads())
print("\n[entrenamiento: s/epoca por dr_train, unrolling]")
spe = {}
for dr in DR_TRAIN:
    ep = 3 if dr <= 120 else 2
    s, _ = t_train_per_epoch(dr, "unrolling", ep)
    spe[("unroll", dr)] = s
    print(f"  dr={dr:4d}: {s:7.3f} s/epoca")

print("\n[entrenamiento: s/epoca implicit]")
s, _ = t_train_per_epoch(None, "implicit", 2)
spe[("implicit", None)] = s
print(f"  implicit: {s:7.3f} s/epoca")

print("\n[evaluacion: costo de UNA pasada de checkpoint (A1+A2, ladder a 8000)]")
e1, wall = t_eval_one_ckpt(5)
print(f"  eval/checkpoint: {e1:7.1f} s   (wall corrida 1ep/1ckpt = {wall:.1f}s)")

print("\n=== RESUMEN (para estimar 400 epocas, 4 checkpoints) ===")
EVAL_TOTAL = 4 * e1
for k, s in spe.items():
    total = 400 * s + EVAL_TOTAL
    print(f"  {str(k):22s}  train400={400*s:8.1f}s + eval={EVAL_TOTAL:7.1f}s = {total:8.1f}s "
          f"({total/60:.1f} min)")
