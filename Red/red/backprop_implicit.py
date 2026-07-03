"""
backprop_implicit.py
====================
Estrategia de backpropagation por DIFERENCIACION IMPLICITA (sin unrolling).

El forward corre Douglas-Rachford hasta convergencia SIN construir grafo
(torch.no_grad): memoria O(1) en el nº de iteraciones. El gradiente NO se obtiene
desenrollando, sino resolviendo el sistema lineal del teorema de la funcion
implicita (IFT) en el punto fijo s* = T(s*, y_hat):

    dy*/dy_hat = C (I - J_s)^{-1} J_y      (J_s = dT/ds,  J_y = dT/dy_hat)

El VJP (backward) resuelve  (I - J_s)^T w = [g; 0]  y devuelve  J_y^T w. Solo se
evalua UNA vez la jacobiana de una iteracion de DR en s* (via torch.func).

Es una estrategia intercambiable: se combina con cualquier arquitectura. La
iteracion de DR del forward es la MISMA que en unrolling (core._dr_iterate); aqui
solo cambia que corre sin grafo y el backward usa la IFT.

Config leida del modelo:  implicit_max_iters, implicit_tol.
"""
import torch


class ImplicitSolver:
    name = "implicit"

    def project(self, model, y_hat, A_poly, B_poly):
        L, c, M_inv = model._dr_precompute(A_poly, B_poly)
        # forward a convergencia SIN grafo (la IFT necesita un punto fijo preciso,
        # no el nº de iteraciones de entrenamiento; es gratis en memoria)
        with torch.no_grad():
            y_k, x_k = model._dr_iterate(y_hat.detach(), L, c, M_inv,
                                         model.implicit_max_iters, tol=model.implicit_tol)
            s_star = torch.cat([y_k, x_k], dim=1)
        y_star = _ImplicitDR.apply(y_hat, s_star, L, c, M_inv, model)
        return model._y_to_matrices(y_star)


class _ImplicitDR(torch.autograd.Function):
    """VJP por el teorema de la funcion implicita (no desenrolla ninguna iteracion)."""
    @staticmethod
    def forward(ctx, y_hat, s_star, L, c, M_inv, model):
        ctx.model = model
        ctx.save_for_backward(y_hat, s_star, L, c, M_inv)
        return s_star[:, :model.dim_y].clone()          # y* = componente-y del punto fijo

    @staticmethod
    def backward(ctx, grad_y):
        from torch.func import vmap, jacrev
        y_hat, s_star, L, c, M_inv = ctx.saved_tensors
        model = ctx.model
        B_sz, d = s_star.shape
        dim_y = model.dim_y
        step = lambda s, yh, L1, c1, Mi: model._dr_step_single(s, yh, L1, c1, Mi)
        J_s = vmap(jacrev(step, argnums=0))(s_star, y_hat, L, c, M_inv)   # (B, d, d)
        J_y = vmap(jacrev(step, argnums=1))(s_star, y_hat, L, c, M_inv)   # (B, d, dim_y)
        I = torch.eye(d, dtype=s_star.dtype, device=s_star.device).expand(B_sz, d, d)
        rhs = torch.zeros(B_sz, d, dtype=s_star.dtype, device=s_star.device)
        rhs[:, :dim_y] = grad_y
        A = (I - J_s).transpose(1, 2) + 1e-8 * I         # ridge por la no-unicidad del punto fijo
        w = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
        grad_yhat = torch.bmm(J_y.transpose(1, 2), w.unsqueeze(-1)).squeeze(-1)
        return grad_yhat, None, None, None, None, None
