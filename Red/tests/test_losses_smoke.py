"""
test_losses_smoke.py  (Parte C — verificación #2)
=================================================
Las 6 pérdidas del registro corren sin NaN/Inf durante 2 épocas sobre ~32 sistemas
reales, con la arquitectura de actuadores. No verifica calidad, solo que el pipeline
entrenamiento+proyección+pérdida es numéricamente sano para cada pérdida.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from entrenamiento.training import load_vertices, train, get_loss_fn, LOSSES_6   # noqa: E402
from red.actuators import LMINetActuators                                        # noqa: E402


def test_losses_smoke():
    items = load_vertices(3, 1, 2, limit=32)
    train_by_N = {2: items}
    for name in LOSSES_6:
        torch.manual_seed(0)
        model = LMINetActuators(n=3, alpha=0.01, dr_iters=30).double()
        loss_fn = get_loss_fn(name, model)
        hist = train(model, train_by_N, epochs=2, lr=1e-3, batch=16,
                     seed=0, log_every=0, loss_fn=loss_fn)
        finite = all(np.isfinite(h) for h in hist)
        print(f"[loss={name:16}] hist={['%.4g' % h for h in hist]}  finite={finite}")
        assert finite, f"pérdida '{name}' produjo NaN/Inf: {hist}"


if __name__ == "__main__":
    test_losses_smoke()
    print("OK  test_losses_smoke")
