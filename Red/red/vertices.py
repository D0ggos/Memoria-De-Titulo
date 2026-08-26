"""
vertices.py
===========
LMI-Net invariante al numero de VERTICES N (encoder Deep Sets, Zaheer et al. 2017).
El numero de actuadores n_u sigue fijo por arquitectura.

Diferencia de arquitectura vs. vanilla:
  - ENTRADA: cada vertice x_i = [vec(A_i), vec(B_i)] pasa por una MLP compartida phi;
    se agrupan con un pooling simetrico (mean/sum/max) -> invariante a permutacion
    y a la CANTIDAD de vertices. Una sola arquitectura sirve para cualquier N.
  - forward(A_poly, B_poly): infiere N de A_poly en cada llamada.

Diferencia vs. actuators: aqui B_i se APLANA dentro del vector por-vertice, asi que
la entrada depende de m (n_u fijo). En actuators cada columna de B es un elemento
de conjunto (invarianza tambien a n_u).
"""
import torch
import torch.nn as nn

from .core import LMICore


class LMINetVertices(LMICore):
    def __init__(self, n=3, m=1, hidden_dim=128, enc_dim=128, alpha=0.1,
                 epsilon=1e-5, dr_iters=100, sigma=0.01, pool="mean", backprop="unrolling",
                 sigma_adaptativo=False):
        super().__init__(n=n, m=m, N=2, alpha=alpha, epsilon=epsilon,
                         dr_iters=dr_iters, sigma=sigma, backprop=backprop,
                         sigma_adaptativo=sigma_adaptativo)  # N=2 placeholder
        assert pool in ("mean", "max", "sum")
        self.pool = pool
        self.vertex_dim = n * n + n * m                     # A_i aplanada + B_i aplanada

        self.phi = nn.Sequential(                           # MLP compartida por vertice
            nn.Linear(self.vertex_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, enc_dim), nn.LayerNorm(enc_dim), nn.Mish(),
        )
        self.rho = nn.Sequential(                           # embedding agregado -> y_hat
            nn.Linear(enc_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, self.dim_y),
        )

    def _encode(self, A_poly, B_poly):
        """A_poly:(B,N,n,n), B_poly:(B,N,n,m) -> y_hat:(B, dim_y). Invariante a N."""
        B_sz, N = A_poly.shape[0], A_poly.shape[1]
        x = torch.cat([A_poly.reshape(B_sz, N, -1), B_poly.reshape(B_sz, N, -1)], dim=-1)
        h = self.phi(x)                                     # (B, N, enc_dim)
        if self.pool == "mean":
            z = h.mean(dim=1)
        elif self.pool == "sum":
            z = h.sum(dim=1)
        else:
            z = h.max(dim=1).values
        return self.rho(z)                                  # (B, dim_y)

    def forward(self, A_poly, B_poly, return_unconstrained=False):
        self.N = A_poly.shape[1]                            # N dinamico para el solver
        y_hat = self._encode(A_poly, B_poly)
        if return_unconstrained:
            return y_hat
        return self._project(y_hat, A_poly, B_poly)
