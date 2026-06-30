"""
lmi_net_deepsets.py
===================
Variante de LMI-Net con un encoder Deep Sets (Zaheer et al., 2017) invariante
a permutacion y a cardinalidad. Reemplaza UNICAMENTE el encoder de entrada;
reutiliza el solver Douglas-Rachford y la cabeza de salida de LMINet sin cambios
(via herencia y el metodo compartido `_project_dr`).

Idea (refleja la matematica del problema):
    entrada = conjunto de vertices {(A_i, B_i)}_{i=1..N}  (tamanio variable)
    salida  = certificado comun (Q, Y)                    (tamanio FIJO = dim_y)

Arquitectura del encoder:
    x_i  = [vec(A_i), vec(B_i)]           in R^{n^2 + n*m}     (por vertice)
    h_i  = phi(x_i)                       in R^{enc_dim}       (MLP compartida)
    z    = pool_i h_i  (mean | max | sum) in R^{enc_dim}       (simetrico => inv. permut.)
    y_hat= rho(z)                         in R^{dim_y}

Como pool no depende de N, una sola arquitectura sirve para cualquier numero de
vertices. La firma de forward cambia a forward(A_poly, B_poly) porque el encoder
ya no consume el vector plano `features`.
"""

import torch
import torch.nn as nn

from .lmi_net import LMINet


class LMINetDeepSets(LMINet):
    def __init__(self, n=3, m=1, hidden_dim=128, enc_dim=128,
                 alpha=0.1, epsilon=1e-5, dr_iters=100, sigma=0.01,
                 pool="mean"):
        # Inicializamos nn.Module directamente para NO construir el self.network
        # de tamanio fijo-en-N de la clase base.
        nn.Module.__init__(self)
        assert pool in ("mean", "max", "sum")
        self.n = n
        self.m = m
        # N es dinamico: se infiere de A_poly en cada forward. Lo dejamos como
        # atributo mutable solo para que _compute_F/_construct_L_c lo lean.
        self.N = None
        self.alpha = alpha
        self.epsilon = epsilon
        self.dr_iters = dr_iters
        self.sigma = sigma
        self.pool = pool

        # Dimensiones de salida (INDEPENDIENTES de N)
        self.dim_Q = (n * (n + 1)) // 2
        self.dim_Y = m * n
        self.dim_y = self.dim_Q + self.dim_Y

        # Caracteristica por vertice: A_i aplanada (n*n) + B_i aplanada (n*m)
        self.vertex_dim = n * n + n * m

        # phi: MLP compartida aplicada a cada vertice por separado
        self.phi = nn.Sequential(
            nn.Linear(self.vertex_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, enc_dim),
            nn.LayerNorm(enc_dim),
            nn.Mish(),
        )
        # rho: procesa el embedding agregado -> y_hat (dim_y)
        self.rho = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.dim_y),
        )

        self.triu_indices = torch.triu_indices(row=self.n, col=self.n, offset=0)

    def _encode(self, A_poly, B_poly):
        """
        A_poly: (B, N, n, n), B_poly: (B, N, n, m) -> y_hat: (B, dim_y).
        Invariante a permutacion de los N vertices y a su cantidad.
        """
        B_sz, N = A_poly.shape[0], A_poly.shape[1]
        # Construir caracteristicas por vertice: (B, N, n*n + n*m)
        a_flat = A_poly.reshape(B_sz, N, -1)
        b_flat = B_poly.reshape(B_sz, N, -1)
        x = torch.cat([a_flat, b_flat], dim=-1)        # (B, N, vertex_dim)

        h = self.phi(x)                                # (B, N, enc_dim)  (MLP compartida)

        if self.pool == "mean":
            z = h.mean(dim=1)
        elif self.pool == "sum":
            z = h.sum(dim=1)
        else:  # max
            z = h.max(dim=1).values                    # (B, enc_dim)

        y_hat = self.rho(z)                            # (B, dim_y)
        return y_hat

    def forward(self, A_poly, B_poly, return_unconstrained=False):
        # N dinamico para que el solver herede el numero de bloques correcto
        self.N = A_poly.shape[1]

        y_hat = self._encode(A_poly, B_poly)
        if return_unconstrained:
            return y_hat

        # Reutiliza EXACTAMENTE el solver Douglas-Rachford de LMINet
        return self._project_dr(y_hat, A_poly, B_poly)
