"""Smoke test de los ejes de sensibilidad (sigma, epsilon, sigma adaptativo).

Comprueba que:
  1. sigma/epsilon llegan efectivamente al modelo cuando la corrida los fija;
  2. omitirlos conserva EXACTAMENTE los defaults de cada arquitectura (vanilla 1e-3,
     invariantes 1e-5), es decir que E1-E4 no cambiaron de comportamiento;
  3. el sigma adaptativo produce un sigma por instancia coherente con la dispersion;
  4. una corrida corta de E7 y otra de E8 entrenan y evaluan de punta a punta.

Uso:  python tests/test_sensibilidad_smoke.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch                                                   # noqa: E402

from barrido.config import RunConfig, build_stage, SIGMAS, EPSILONS   # noqa: E402
from barrido.run import build_model, run_one                   # noqa: E402
from barrido.data import load_cell, stack_cell                 # noqa: E402

torch.set_default_dtype(torch.float64)
OUT = ROOT / "analisis" / "resultados" / "barrido" / "_smoke_sens"


def test_ejes_llegan_al_modelo():
    c = RunConfig(stage="T", loss="paper", sigma=0.1, epsilon=1e-2)
    m = build_model(c)
    assert m.sigma == 0.1, m.sigma
    assert m.epsilon == 1e-2, m.epsilon
    print("  [1] sigma/epsilon llegan al modelo: OK")


def test_defaults_intactos():
    """Sin fijarlos, cada arquitectura conserva SU default (no uno global)."""
    esperado = {"actuadores": 1e-5, "vertices": 1e-5, "vanilla": 1e-3}
    for arch, eps in esperado.items():
        m = build_model(RunConfig(stage="T", loss="paper", arch=arch))
        assert m.epsilon == eps, f"{arch}: {m.epsilon} != {eps}"
        assert m.sigma == 0.01, f"{arch}: sigma {m.sigma}"
        assert not m.sigma_adaptativo
    print("  [2] defaults por arquitectura intactos (vanilla 1e-3, invariantes 1e-5): OK")


def test_sigma_adaptativo():
    c = RunConfig(stage="T", loss="paper", n_x=3, sigma=0.01, sigma_adaptativo=True)
    m = build_model(c)
    items = load_cell(3, 1, 2)[:8]
    A, B = stack_cell(items)
    _ = m._dr_precompute(A, B)
    assert torch.is_tensor(m.sigma), "sigma deberia ser un tensor por instancia"
    assert m.sigma.shape == (A.shape[0], 1), m.sigma.shape
    assert (m.sigma > 0).all() and (m.sigma <= m.sigma_base).all()
    disp = torch.linalg.matrix_norm(A - A.mean(dim=1, keepdim=True),
                                    ord=2, dim=(-2, -1)).amax(dim=1)
    # mas dispersion -> menos anclaje: la correlacion debe ser negativa
    s = m.sigma.squeeze(1)
    corr = torch.corrcoef(torch.stack([disp, s]))[0, 1].item()
    assert corr < -0.9, f"correlacion dispersion-sigma = {corr:.3f}"
    # el forward completo sigue funcionando con sigma vectorial
    Q, Y = m(A, B)
    assert torch.isfinite(Q).all() and torch.isfinite(Y).all()
    print(f"  [3] sigma adaptativo: tensor {tuple(m.sigma.shape)}, "
          f"corr(dispersion, sigma) = {corr:.3f}, forward finito: OK")


def test_corridas_cortas():
    """Una celda de E7 y una de E8, en version rapida, de punta a punta."""
    base = dict(loss="paper", arch="actuadores", backprop="unrolling", dr_train=30,
                alpha=0.01, n_x=3, train_size=50, seed=42,
                epochs=2, epoch_ckpts=(1, 2), dr_eval=(100, 500))
    casos = [RunConfig(stage="E7", sigma=1.0, epsilon=1e-2, **base),
             RunConfig(stage="E8", sigma=0.01, sigma_adaptativo=True, **base)]
    for c in casos:
        res = run_one(c, OUT / c.stage)
        assert res.get("status") == "ok", f"{c.run_id()} -> {res}"
        assert res.get("n_rows", 0) > 0, res
        print(f"  [4] {c.stage} corre completo: {res['n_rows']:,} filas, "
              f"{res['t_train_s']:.1f} s  ({c.run_id()[:52]}...)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print(f"malla de sensibilidad: sigma={SIGMAS}  epsilon={EPSILONS}")
    print(f"E7 = {len(build_stage('E7'))} corridas   E8 = {len(build_stage('E8'))} corridas\n")
    test_ejes_llegan_al_modelo()
    test_defaults_intactos()
    test_sigma_adaptativo()
    test_corridas_cortas()
    print("\nOK  test_sensibilidad_smoke")
