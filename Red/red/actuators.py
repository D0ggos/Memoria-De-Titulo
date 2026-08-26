"""
actuators.py
============
LMI-Net invariante a N vertices Y equivariante al numero de ACTUADORES n_u.

Diferencia de arquitectura vs. vertices (el cambio que da la invarianza a n_u):
  - ENTRADA: en vez de aplanar B_i, cada columna b_ij (un actuador) es un elemento
    de conjunto. Token por (vertice i, actuador j):  t_ij = [vec(A_i), b_ij] in R^{n^2+n}
    -> el tamaño de entrada NO depende de m ni de N.
  - POOLING en dos ejes: pool sobre vertices -> embedding por actuador u_j;
    pool sobre actuadores -> embedding global z.
  - SALIDA (cabeza partida):
      rho_Q(z)         -> Q   (invariante, tamaño fijo)
      rho_Y([u_j, z])  -> Y_j (una fila por actuador -> EQUIVARIANTE a n_u)
  - forward(A_poly, B_poly): infiere N y m; el solver hereda las formas correctas.

Una sola arquitectura sirve para cualquier (N, n_u). El solver DR viene de LMICore.
"""
import torch
import torch.nn as nn

from .core import LMICore


class LMINetActuators(LMICore):
    def __init__(self, n=3, hidden_dim=128, enc_dim=128, alpha=0.1,
                 epsilon=1e-5, dr_iters=100, sigma=0.01, pool="mean", backprop="unrolling",
                 sigma_adaptativo=False):
        super().__init__(n=n, m=1, N=2, alpha=alpha, epsilon=epsilon,
                         dr_iters=dr_iters, sigma=sigma, backprop=backprop,
                         sigma_adaptativo=sigma_adaptativo)  # m,N,dim_y dinamicos
        assert pool in ("mean", "max", "sum")
        self.pool = pool
        self.token_dim = n * n + n                          # [vec(A_i), b_ij]  (sin m ni N)

        self.phi = nn.Sequential(
            nn.Linear(self.token_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, enc_dim), nn.LayerNorm(enc_dim), nn.Mish(),
        )
        self.rho_Q = nn.Sequential(                         # Q: invariante
            nn.Linear(enc_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, self.dim_Q),
        )
        self.rho_Y = nn.Sequential(                         # una fila de Y por actuador
            nn.Linear(2 * enc_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, n),
        )

    def _pool(self, h, dim):
        if self.pool == "mean":
            return h.mean(dim=dim)
        if self.pool == "sum":
            return h.sum(dim=dim)
        return h.max(dim=dim).values

    def _encode(self, A_poly, B_poly):
        """A_poly:(B,N,n,n), B_poly:(B,N,n,m) -> y_hat:(B, dim_Q + m*n)."""
        Bsz, N, n = A_poly.shape[0], A_poly.shape[1], self.n
        m = B_poly.shape[-1]
        a = A_poly.reshape(Bsz, N, 1, n * n).expand(-1, -1, m, -1)   # (B,N,m,n^2)
        b = B_poly.permute(0, 1, 3, 2)                               # (B,N,m,n)
        t = torch.cat([a, b], dim=-1)                                # token por (i,j)
        h = self.phi(t)                                              # (B,N,m,enc)
        u = self._pool(h, dim=1)                                     # (B,m,enc) por actuador
        z = self._pool(u, dim=1)                                     # (B,enc)   global
        vec_Q = self.rho_Q(z)
        z_rep = z.unsqueeze(1).expand(-1, m, -1)
        vec_Y = self.rho_Y(torch.cat([u, z_rep], dim=-1))           # (B,m,n)
        return torch.cat([vec_Q, vec_Y.reshape(Bsz, -1)], dim=-1)

    def forward(self, A_poly, B_poly, return_unconstrained=False):
        self.N = A_poly.shape[1]
        self.m = B_poly.shape[-1]
        self.dim_Y = self.m * self.n
        self.dim_y = self.dim_Q + self.dim_Y
        y_hat = self._encode(A_poly, B_poly)
        if return_unconstrained:
            return y_hat
        return self._project(y_hat, A_poly, B_poly)
