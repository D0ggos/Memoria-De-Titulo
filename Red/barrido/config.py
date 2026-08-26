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
# Ejes del analisis de sensibilidad (E7). Se incluyen los defaults del codigo (sigma=0.01,
# epsilon=1e-5) para que el barrido conecte con los resultados ya existentes.
# 0.005 se agrego despues de ver que 0.01 ganaba en el BORDE de la grilla original
# {0.01, 0.1, 1.0}: hacia falta un punto por debajo para saber si el optimo es interior.
SIGMAS = [0.005, 0.01, 0.1, 1.0]              # peso de anclaje a la propuesta del encoder
EPSILONS = [1e-5, 1e-4, 1e-3, 1e-2]           # margen de positividad  Q >= epsilon*I

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
    # Ejes de sensibilidad. None = usar el default de la arquitectura (sigma=0.01 en las
    # tres; epsilon=1e-5 en actuadores/vertices y 1e-3 en vanilla, fiel al paper). Se dejan
    # en None por defecto para que run_id() NO cambie en las corridas ya hechas.
    sigma: Optional[float] = None
    epsilon: Optional[float] = None
    sigma_adaptativo: bool = False
    # opciones de evaluacion (no afectan el entrenamiento)
    cvxpy_ceiling: bool = False               # calcular techo CVXPY en A1/A2 (lento)
    cvxpy_max_systems: int = 30               # tope de sistemas por celda para CVXPY
    per_system: bool = True                   # guardar filas por-sistema (crudo) ademas del agregado

    def run_id(self) -> str:
        a = f"{self.alpha:g}".replace(".", "p")
        dr = "impl" if self.backprop == "implicit" else f"dr{self.dr_train}"
        base = (f"{self.stage}__{self.arch}__nx{self.n_x}__{self.loss}__{self.backprop}"
                f"__{dr}__a{a}__sz{self.train_size}__s{self.seed}")
        # Los ejes de sensibilidad solo se anexan cuando se apartan del default: asi los
        # run_id de las corridas E1-E4 ya ejecutadas siguen siendo bit a bit los mismos y
        # el `resume` del driver las sigue reconociendo.
        extra = ""
        if self.sigma is not None:
            extra += f"__sg{self.sigma:g}".replace(".", "p")
        if self.epsilon is not None:
            extra += f"__eps{self.epsilon:g}".replace(".", "p").replace("-", "m")
        if self.sigma_adaptativo:
            extra += "__sgadapt"
        return base + extra

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
    """Retropropagacion + escalera de arquitectura x orden:
       (a) perdida(6) x {unrolling(dr=30), implicit} x semilla(2), en la config base;
       (b) perdida(6) x arquitectura NO base(2: vertices, vanilla) x n_x(4) x semilla(2).
    Las celdas de `actuadores` para n_x != 3 las aporta E2, que barre el orden con esa
    arquitectura; juntas completan la malla arquitectura x orden sin repetir corridas.
    Resto en base (size=150, alpha=0.01, dr=30). vanilla no es invariante -> no tiene A2.
    120 corridas."""
    out = []
    for loss in LOSSES:
        for seed in SEEDS:
            # (a) duelo de retropropagacion, config base completa
            for bp, dr in (("unrolling", BASE_DR_TRAIN), ("implicit", None)):
                out.append(RunConfig(stage="E4", loss=loss, arch=BASE_ARCH, backprop=bp,
                                     dr_train=dr, alpha=BASE_ALPHA, n_x=BASE_NX,
                                     train_size=BASE_SIZE, seed=seed))
            # (b) escalera de arquitectura en TODOS los ordenes
            for nx in NX:
                for arch in ARCHS:
                    if arch == BASE_ARCH:
                        continue                          # n_x=3 ya esta en (a); el resto lo da E2
                    out.append(RunConfig(stage="E4", loss=loss, arch=arch,
                                         backprop="unrolling", dr_train=BASE_DR_TRAIN,
                                         alpha=BASE_ALPHA, n_x=nx, train_size=BASE_SIZE,
                                         seed=seed))
    return out


def stage_E7():
    """Sensibilidad a sigma (anclaje) y epsilon (margen de positividad).

    Los dos se fijaban globalmente y nunca se barrieron, pese a que epsilon difiere por
    arquitectura (1e-3 en vanilla, 1e-5 en las invariantes) y sigma gradua cuanta
    influencia tiene el codificador sobre la proyeccion. Malla completa
    sigma(3) x epsilon(4) x perdida(6) x semilla(2) = 144 corridas, resto en configuracion
    base (actuadores, n_x=3, unrolling dr=30, alpha=0.01, size=150).

    Prediccion a contrastar: la LMI es homogenea de grado 1 en (Q,Y), de modo que epsilon
    solo fija la ESCALA del certificado y no altera que sistemas son certificables ni la
    ganancia K = Y Q^-1. Deberia entonces ser inocuo para las perdidas invariantes de
    escala y no para las demas.

    (E5 es la evaluacion OOD, que no entrena; E6 se fusiono dentro de E4. De ahi el salto.)
    """
    return [RunConfig(stage="E7", loss=loss, arch=BASE_ARCH, backprop="unrolling",
                      dr_train=BASE_DR_TRAIN, alpha=BASE_ALPHA, n_x=BASE_NX,
                      train_size=BASE_SIZE, seed=seed, sigma=sg, epsilon=eps)
            for loss in LOSSES for sg in SIGMAS for eps in EPSILONS for seed in SEEDS]


def stage_E8():
    """Sigma adaptativo: sigma_i = sigma_base / (1 + dispersion del politopo).

    Explora la sugerencia de modular el anclaje segun la geometria de cada instancia, en
    los ordenes altos donde las celdas son mas dificiles. Se compara contra la misma
    configuracion con sigma fijo, que ya aporta E2 (n_x) y E7 (sigma=0.01 nominal).
    perdida(6) x n_x(2: 4 y 5) x semilla(2) = 24 corridas."""
    return [RunConfig(stage="E8", loss=loss, arch=BASE_ARCH, backprop="unrolling",
                      dr_train=BASE_DR_TRAIN, alpha=BASE_ALPHA, n_x=nx,
                      train_size=BASE_SIZE, seed=seed, sigma=0.01, sigma_adaptativo=True)
            for loss in LOSSES for nx in (4, 5) for seed in SEEDS]


def stage_E9():
    """Control del confound de epsilon en el contraste de arquitecturas.

    E4 compara vanilla (epsilon=1e-3, fiel al paper) contra vertices y actuadores
    (epsilon=1e-5): la ventaja de vanilla dentro de distribucion esta confundida con dos
    ordenes de magnitud de diferencia en epsilon. Aqui se reentrenan vanilla y vertices en
    AMBOS valores para poder comparar a epsilon igualado; las celdas de actuadores las
    aporta E7 (sigma=0.01, epsilon en la grilla). arquitectura(2) x epsilon(2) x
    perdida(6) x semilla(2) = 48 corridas, resto en configuracion base."""
    return [RunConfig(stage="E9", loss=loss, arch=arch, backprop="unrolling",
                      dr_train=BASE_DR_TRAIN, alpha=BASE_ALPHA, n_x=BASE_NX,
                      train_size=BASE_SIZE, seed=seed, sigma=0.01, epsilon=eps)
            for loss in LOSSES for arch in ("vanilla", "vertices")
            for eps in (1e-5, 1e-3) for seed in SEEDS]


STAGES = {"E0": stage_E0, "E1": stage_E1, "E2": stage_E2, "E3": stage_E3, "E4": stage_E4,
          "E7": stage_E7, "E8": stage_E8, "E9": stage_E9}
# E5 (OOD) NO entrena: solo evalua modelos de E1-E4 (ver barrido/ood_eval y Parte E).


def build_stage(name: str):
    if name not in STAGES:
        raise ValueError(f"etapa desconocida: {name}. Disponibles: {list(STAGES)} (E5 es solo-eval).")
    return STAGES[name]()
