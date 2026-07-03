"""
vanilla.py
==========
LMI-Net VANILLA — reproduccion FIEL de la receta del apendice del paper LMI-Net.
Es el baseline contra el que se comparan las contribuciones de la tesis
(arquitecturas vertices/actuators, estrategia unrolling).

Receta del paper (defaults del constructor):
  - Backbone: MLP de dos capas ocultas (in -> 64 -> 64 -> out), 64 neuronas por capa,
    activacion ReLU, SIN LayerNorm.  ("two-layer MLP backbone with 64 neurons per
    layer and ReLU activations").
  - Backward: diferenciacion IMPLICITA (Sec. III.D del paper). El paper NO corre hasta
    convergencia: usa n_iter = 500 iteraciones FIJAS y evalua la IFT en z_{500}.
    Por eso: dr_iters=500, implicit_max_iters=500, implicit_tol=0.0 (sin early-stop).
  - Hiperparametros: epsilon=1e-3 (Ec. 23), sigma=0.01, float64.

Notas:
  - `alpha` (tasa de decaimiento) NO es un hiperparametro del paper (su LMI es LTI con
    canal B_w, sin decay); es la adaptacion de este repo a la LMI de estabilizacion
    robusta politopica. Se fija en 0.01 para coincidir con el resto del repo y el
    subconjunto RS de reproduccion.
  - `hidden_dim`, `n_hidden` y `activation` quedan parametrizables para ablation, pero
    los DEFAULTS son los del paper.
  - `unrolling` sigue disponible con backprop="unrolling" (variante experimental de la
    tesis, no del paper).

Diferencia de arquitectura vs. las otras: entrada = vector aplanado de todos los
vertices, tamaño N(n^2 + n·m) -> NO invariante (una arquitectura por (n_x, n_u, N)).
"""
import torch
import torch.nn as nn

from .core import LMICore

_ACTIVATIONS = {"relu": nn.ReLU, "mish": nn.Mish, "tanh": nn.Tanh, "gelu": nn.GELU}


class LMINetVanilla(LMICore):
    def __init__(self, n=3, m=1, N=2, hidden_dim=64, n_hidden=2, activation="relu",
                 alpha=0.01, epsilon=1e-3, dr_iters=500, sigma=0.01, backprop="implicit"):
        super().__init__(n=n, m=m, N=N, alpha=alpha, epsilon=epsilon,
                         dr_iters=dr_iters, sigma=sigma, backprop=backprop)
        # Regimen de iteraciones fiel al paper: 500 iteraciones DR FIJAS (sin early-stop).
        self.implicit_max_iters = dr_iters
        self.implicit_tol = 0.0

        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation '{activation}' no reconocida: {list(_ACTIVATIONS)}")
        act = _ACTIVATIONS[activation]

        input_dim = N * (n * n) + N * (n * m)      # depende de N, n, m => NO invariante
        layers, d = [], input_dim
        for _ in range(n_hidden):                  # in -> 64 -> 64  (ReLU, sin LayerNorm)
            layers += [nn.Linear(d, hidden_dim), act()]
            d = hidden_dim
        layers += [nn.Linear(d, self.dim_y)]       # -> salida y
        self.network = nn.Sequential(*layers)

    def forward(self, A_poly, B_poly, return_unconstrained=False):
        B_sz = A_poly.shape[0]
        features = torch.cat([A_poly.reshape(B_sz, -1), B_poly.reshape(B_sz, -1)], dim=1)
        y_hat = self.network(features)
        if return_unconstrained:
            return y_hat
        return self._project(y_hat, A_poly, B_poly)
