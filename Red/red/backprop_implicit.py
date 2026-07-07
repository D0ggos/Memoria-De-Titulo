"""
backprop_implicit.py
====================
Diferenciacion IMPLICITA (sin unrolling), con backward MATRIX-FREE.

FORWARD
  Corre Douglas-Rachford hasta el punto fijo  s* = T(s*, y_hat)  SIN construir grafo
  (memoria O(1) en el nº de iteraciones) y aplica la PROYECCION FINAL sobre el
  subespacio afin C1 para definir la salida:
        y* = Pi_{C1}(y_hat, s*) = _dr_final_proj(y_hat, y_k*, x_k*)
  IGUAL que la rama de unrolling (salida UNIFICADA, Parte A). Esa proyeccion final
  queda DENTRO de la funcion diferenciada (_ImplicitDR), asi que el VJP la atraviesa.

BACKWARD  (teorema de la funcion implicita, IFT)
  El punto fijo s* depende de y_hat; derivando s* = T(s*, y_hat):
        ds*/dy_hat = (I - J_s)^{-1} J_y ,   J_s = dT/ds ,  J_y = dT/dy_hat  (en s*).
  Con la salida y* = g(y_hat, s*) (g = proyeccion final), la regla de la cadena da:
        dy*/dy_hat = dg/dy_hat  +  dg/ds* . (I - J_s)^{-1} J_y .
  El adjunto (VJP) NO materializa ninguna Jacobiana: solo pide PRODUCTOS
  vector-Jacobiana via torch.func.vjp sobre UNA iteracion de DR (core._dr_state_step)
  y sobre la proyeccion final (core._dr_final_proj). El sistema lineal
        (I - J_s)^T w = rhs
  se resuelve por iteracion de punto fijo estilo DEQ:
        w <- (J_s^T w + rhs) / (1 + ridge),    cada paso = UN VJP,
  donde el ridge (I - J_s + eps·I) regulariza la no-unicidad del punto fijo del DR
  (sesgo despreciable, estandar en DEQ). Esto reemplaza al jacrev anterior, que
  construia J_s in R^{dxd} (d = dim_y + (N+1)·n^2) con ~d pases de backward por
  muestra: era el cuello de botella del barrido (la rama implicita ~70% del computo).

Config leida del modelo:
  forward :  implicit_max_iters, implicit_tol.
  backward:  implicit_adjoint_iters, implicit_adjoint_tol, implicit_ridge,
             implicit_diag_monitor.
"""
import torch


class ImplicitSolver:
    name = "implicit"

    def project(self, model, y_hat, A_poly, B_poly):
        L, c, M_inv = model._dr_precompute(A_poly, B_poly)
        # forward a convergencia SIN grafo: la IFT necesita un punto fijo preciso,
        # no el nº de iteraciones de entrenamiento (gratis en memoria).
        with torch.no_grad():
            y_k, x_k = model._dr_iterate(y_hat.detach(), L, c, M_inv,
                                         model.implicit_max_iters, tol=model.implicit_tol)
            s_star = torch.cat([y_k, x_k], dim=1)
        y_star = _ImplicitDR.apply(y_hat, s_star, L, c, M_inv, model)
        return model._y_to_matrices(y_star)


# --------------------------- utilidades matrix-free -------------------------
def _adjoint_solve(matvec_JsT, rhs, iters, tol, ridge):
    """Resuelve  (I - J_s + ridge·I)^T w = rhs  =>  (I - J_s^T + ridge·I) w = rhs
    por iteracion de punto fijo:   w <- (J_s^T w + rhs) / (1 + ridge).
    `matvec_JsT(v)` = J_s^T v (un VJP). Para en ||Δw||_inf < tol.
    Devuelve (w, iters_usadas, residual_final)."""
    w = rhs.clone()
    denom = 1.0 + ridge
    used, res = 0, float("inf")
    for k in range(iters):
        w_new = (matvec_JsT(w) + rhs) / denom
        res = (w_new - w).abs().max().item()
        w = w_new
        used = k + 1
        if res < tol:
            break
    return w, used, res


@torch.no_grad()
def _spectral_gap_min(model, s_star, y_hat, L, c, M_inv):
    """Diagnostico (no aborta): min_{i!=j} |lam_i - lam_j| sobre los N+1 bloques
    simetricos reflejados (xin = 2·x_w - x_k) que entran a eigh en la iteracion de
    DR evaluada en s*. Un gap chico avisa que el gradiente espectral (backward de
    eigh) esta mal condicionado en esa muestra."""
    B_sz = y_hat.shape[0]; block = model.n * model.n; nb = model.N + 1
    Lt = L.transpose(1, 2); s2 = 2 * model.sigma
    y_k = s_star[:, :model.dim_y]; x_k = s_star[:, model.dim_y:]
    y_avg = (s2 * y_hat + y_k) / (s2 + 1.0)
    term2 = torch.bmm(Lt, (c - x_k).unsqueeze(-1)).squeeze(-1)
    y_w = torch.bmm(M_inv, (y_avg - term2).unsqueeze(-1)).squeeze(-1)
    x_w = torch.bmm(L, y_w.unsqueeze(-1)).squeeze(-1) + c
    xin = 2 * x_w - x_k
    gmin = float("inf")
    for b in range(nb):
        X = xin[:, b * block:(b + 1) * block].view(B_sz, model.n, model.n)
        X = 0.5 * (X + X.transpose(1, 2))
        lam = torch.linalg.eigvalsh(X)                       # (B, n) ascendente
        gaps = (lam[:, 1:] - lam[:, :-1])                    # (B, n-1) >= 0
        if gaps.numel():
            gmin = min(gmin, gaps.min().item())
    return gmin


class _ImplicitDR(torch.autograd.Function):
    """VJP por el teorema de la funcion implicita, MATRIX-FREE. La salida y* incluye la
    proyeccion final Pi_{C1}, que el backward atraviesa (via el termino dg/dy_hat directo
    y el termino dg/ds* propagado por el adjunto de la IFT)."""

    @staticmethod
    def forward(ctx, y_hat, s_star, L, c, M_inv, model):
        ctx.model = model
        ctx.save_for_backward(y_hat, s_star, L, c, M_inv)
        y_k = s_star[:, :model.dim_y]; x_k = s_star[:, model.dim_y:]
        return model._dr_final_proj(y_hat, y_k, x_k, L, c, M_inv)      # y* = Pi_{C1}(y_hat, s*)

    @staticmethod
    def backward(ctx, grad_out):
        y_hat, s_star, L, c, M_inv = ctx.saved_tensors
        model = ctx.model
        dim_y = model.dim_y

        # (1) VJP de la proyeccion final g(y_hat, s*): reparte grad_out en sus dos vias.
        def g_fn(yh, s):
            yk = s[:, :dim_y]; xk = s[:, dim_y:]
            return model._dr_final_proj(yh, yk, xk, L, c, M_inv)
        _, vjp_g = torch.func.vjp(g_fn, y_hat, s_star)
        g_yhat, g_s = vjp_g(grad_out)                # dg/dy_hat^T grad_out ,  dg/ds*^T grad_out

        # (2) adjunto matrix-free: (I - J_s)^T w = g_s, sin materializar J_s.
        #     J_s^T v  y  J_y^T v  son VJPs de UNA iteracion de DR evaluada en s*.
        _, vjp_s = torch.func.vjp(lambda s: model._dr_state_step(s, y_hat, L, c, M_inv), s_star)
        _, vjp_y = torch.func.vjp(lambda yh: model._dr_state_step(s_star, yh, L, c, M_inv), y_hat)
        w, used, res = _adjoint_solve(lambda v: vjp_s(v)[0], g_s, model.implicit_adjoint_iters,
                                      model.implicit_adjoint_tol, model.implicit_ridge)
        if getattr(model, "implicit_diag_monitor", False):
            gmin = _spectral_gap_min(model, s_star, y_hat, L, c, M_inv)
            print(f"[implicit-diag] gap_espectral_min={gmin:.3e}  adjunto: iters={used} res={res:.2e}")

        # (3) grad_yhat = dg/dy_hat^T grad_out  +  J_y^T w
        grad_yhat = g_yhat + vjp_y(w)[0]
        return grad_yhat, None, None, None, None, None
