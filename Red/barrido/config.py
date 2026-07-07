"""
config.py  —  ejes, RunConfig y generadores de etapas (Parte D)
===============================================================
Una CORRIDA = una `RunConfig` completa (todos los ejes fijados). El barrido es una
LISTA de RunConfig; el driver las reparte en procesos. Epocas y dr_eval NO son ejes de
corrida: se obtienen por CHECKPOINT dentro de UNA sola corrida (una de 400 epocas ->
{50,100,200,400}; una sola pasada de evaluacion -> ladder de dr_eval).

Las 6 perdidas se barren COMPLETAS en todas las etapas (factorial balanceado, sin
seleccion adaptativa de "finalistas").
"""
from dataclasses import dataclass, field, asdict
from typing import Optional

from entrenamiento.training import LOSSES_6

# --------------------------- Ejes del barrido ------------------------------
LOSSES = list(LOSSES_6)                       # 6 perdidas (Parte C)
ARCHS = ["vanilla", "vertices", "actuadores"]
ALPHAS = [0.001, 0.01, 0.05, 0.1]
DR_TRAIN = [5, 30, 120, 500]                  # solo para unrolling (implicit no tiene este eje)
NX = [2, 3, 4, 5]
SIZES = [50, 150, 400]                        # tamaño de train por celda
SEEDS = [42, 123]

EPOCHS = 400                                  # corrida larga; se leen checkpoints intermedios
EPOCH_CKPTS = (50, 100, 200, 400)             # checkpoints de epoca (no multiplican entrenamientos)
DR_EVAL = (100, 250, 500, 1000, 2000, 4000, 8000)   # ladder de dr_eval (una pasada con checkpoints)

# --------------------------- Config base -----------------------------------
# Ejes NO barridos por E2-E4 se fijan aqui (salvo el eje que cada etapa varia).
BASE_ARCH = "actuadores"
BASE_NX = 3
BASE_SIZE = 150
BASE_DR_TRAIN = 30
BASE_ALPHA = 0.01
BASE_N = 2                                     # N unico para arquitecturas NO invariantes (vanilla)
BASE_M = 1


@dataclass(frozen=True)
class RunConfig:
    """Config completa de UNA corrida (un entrenamiento). run_id() es determinista y sirve
    de nombre de shard (CSV/parquet) y carpeta de checkpoints del modelo."""
    stage: str
    loss: str
    arch: str = BASE_ARCH
    backprop: str = "unrolling"               # "unrolling" | "implicit"
    dr_train: Optional[int] = BASE_DR_TRAIN   # None si implicit
    alpha: float = BASE_ALPHA
    n_x: int = BASE_NX
    train_size: int = BASE_SIZE
    seed: int = 42
    epochs: int = EPOCHS
    epoch_ckpts: tuple = EPOCH_CKPTS
    dr_eval: tuple = DR_EVAL
    # opciones de evaluacion (no afectan el entrenamiento)
    cvxpy_ceiling: bool = False               # calcular techo CVXPY en A1/A2 (lento)
    cvxpy_max_systems: int = 30               # tope de sistemas por celda para CVXPY
    per_system: bool = True                   # guardar filas por-sistema (crudo) ademas del agregado

    def run_id(self) -> str:
        a = f"{self.alpha:g}".replace(".", "p")
        dr = "impl" if self.backprop == "implicit" else f"dr{self.dr_train}"
        return (f"{self.stage}__{self.arch}__nx{self.n_x}__{self.loss}__{self.backprop}"
                f"__{dr}__a{a}__sz{self.train_size}__s{self.seed}")

    def as_row(self) -> dict:
        """Columnas de config que se replican en cada fila de resultado (para poder
        filtrar/pivotar el CSV crudo sin joins)."""
        d = asdict(self)
        d.pop("epoch_ckpts"); d.pop("dr_eval")
        d["run_id"] = self.run_id()
        return d


# --------------------------- Generadores de etapas -------------------------
def stage_E0():
    """Sanidad: config base (actuadores, n_x=3) x 2 semillas, en version RAPIDA (pocas
    epocas, ladder corto) para smoke-testear el pipeline entero de punta a punta."""
    out = []
    for seed in SEEDS:
        for bp, dr in [("unrolling", BASE_DR_TRAIN), ("implicit", None)]:
            out.append(RunConfig(
                stage="E0", loss="paper", arch=BASE_ARCH, backprop=bp, dr_train=dr,
                alpha=BASE_ALPHA, n_x=BASE_NX, train_size=50, seed=seed,
                epochs=2, epoch_ckpts=(1, 2), dr_eval=(100, 500),
                cvxpy_ceiling=False, per_system=True))
    return out


def stage_E1():
    """Factorial nucleo: perdida(6) x alpha(4) x dr_train(4) x semilla(2), arquitectura
    actuadores, n_x=3, tamaño=150, checkpoints de epoca. Rama implicita: MISMOS ejes SIN
    dr_train (perdida x alpha x semilla)."""
    out = []
    for loss in LOSSES:
        for alpha in ALPHAS:
            for seed in SEEDS:
                for dr in DR_TRAIN:                       # unrolling
                    out.append(RunConfig(stage="E1", loss=loss, arch="actuadores",
                                         backprop="unrolling", dr_train=dr, alpha=alpha,
                                         n_x=3, train_size=150, seed=seed))
                out.append(RunConfig(stage="E1", loss=loss, arch="actuadores",   # implicit
                                     backprop="implicit", dr_train=None, alpha=alpha,
                                     n_x=3, train_size=150, seed=seed))
    return out


def stage_E2():
    """Orden: perdida(6) x n_x(4) x semilla(2). Resto en base (actuadores, size=150,
    unrolling dr=30, alpha=0.01)."""
    return [RunConfig(stage="E2", loss=loss, arch=BASE_ARCH, backprop="unrolling",
                      dr_train=BASE_DR_TRAIN, alpha=BASE_ALPHA, n_x=nx,
                      train_size=BASE_SIZE, seed=seed)
            for loss in LOSSES for nx in NX for seed in SEEDS]


def stage_E3():
    """Curva de datos: perdida(6) x tamaño(3) x semilla(2). Resto en base."""
    return [RunConfig(stage="E3", loss=loss, arch=BASE_ARCH, backprop="unrolling",
                      dr_train=BASE_DR_TRAIN, alpha=BASE_ALPHA, n_x=BASE_NX,
                      train_size=sz, seed=seed)
            for loss in LOSSES for sz in SIZES for seed in SEEDS]


def stage_E4():
    """Backprop + escalera de arquitectura:
       (a) perdida(6) x {unrolling(dr=30), implicit} x semilla(2);
       (b) perdida(6) x arquitectura(3) x semilla(2).
    Resto en base (n_x=3, size=150, alpha=0.01)."""
    out = []
    for loss in LOSSES:
        for seed in SEEDS:
            out.append(RunConfig(stage="E4", loss=loss, arch=BASE_ARCH, backprop="unrolling",
                                 dr_train=BASE_DR_TRAIN, alpha=BASE_ALPHA, n_x=BASE_NX,
                                 train_size=BASE_SIZE, seed=seed))
            out.append(RunConfig(stage="E4", loss=loss, arch=BASE_ARCH, backprop="implicit",
                                 dr_train=None, alpha=BASE_ALPHA, n_x=BASE_NX,
                                 train_size=BASE_SIZE, seed=seed))
            for arch in ARCHS:
                if arch == BASE_ARCH:
                    continue                              # ya cubierto por la fila unrolling de arriba
                out.append(RunConfig(stage="E4", loss=loss, arch=arch, backprop="unrolling",
                                     dr_train=BASE_DR_TRAIN, alpha=BASE_ALPHA, n_x=BASE_NX,
                                     train_size=BASE_SIZE, seed=seed))
    return out


STAGES = {"E0": stage_E0, "E1": stage_E1, "E2": stage_E2, "E3": stage_E3, "E4": stage_E4}
# E5 (OOD) NO entrena: solo evalua modelos de E1-E4 (ver barrido/ood_eval y Parte E).


def build_stage(name: str):
    if name not in STAGES:
        raise ValueError(f"etapa desconocida: {name}. Disponibles: {list(STAGES)} (E5 es solo-eval).")
    return STAGES[name]()
