"""
core.py
=======
Nucleo COMPARTIDO por las tres arquitecturas LMI-Net y por las dos estrategias de
backpropagation. Aisla lo que es identico en todas para que las diferencias queden
en archivos separados:

  arquitectura (encoder)   -> vanilla.py / vertices.py / actuators.py
  backpropagation          -> backprop_unrolling.py / backprop_implicit.py

Aqui vive:
  - construccion de la LMI robusta            (_compute_F, _construct_L_c)
  - decodificacion  y -> (Q, Y)               (_y_to_matrices, _vec_to_Q)
  - el forward de Douglas-Rachford COMPARTIDO  (_dr_precompute, _dr_step_batch,
    _dr_iterate, _dr_final_proj, _dr_state_step, _dr_step_single)
    <- lo mismo para unrolling e implicita
  - la SALIDA UNIFICADA de ambos solvers: los dos aterrizan con la proyeccion final
    Pi_{C1} (_dr_final_proj) antes de decodificar (Q, Y). El paper define
    y* = Pi_{C1}(z_niter); C1 es la restriccion afin que codifica la dinamica de la
    planta, asi que proyectar ahi garantiza que (Q,Y) respeta esa estructura exacta.
    Unificar aisla el gradiente como UNICA variable entre unrolling e implicita.
  - la seleccion de la estrategia de backprop  (self.solver)

Se puede instanciar directamente como "solver sin encoder" (lo usan
validate_projection.py y el benchmark de proyeccion).
"""
import torch
import torch.nn as nn

from .backprop_unrolling import UnrollingSolver
from .backprop_implicit import ImplicitSolver


def lmi_blocks(Q, Y, A_poly, B_poly, alpha, epsilon):
    """Bloques F_i(y) de la LMI (N+1 bloques n×n simetricos) desde (Q, Y):
        F_i     = -(A_i Q + Q A_iᵀ + B_i Y + Yᵀ B_iᵀ + 2·alpha·Q),  i = 1..N
        F_{N+1} =  Q - epsilon·I
    Helper COMPARTIDO por el solver (core._compute_F) y la perdida del paper
    (training.paper_loss), para no duplicar la construccion de F.
    A_poly:(B,N,n,n)  B_poly:(B,N,n,m)  Q:(B,n,n)  Y:(B,m,n)."""
    B_sz, n = Q.shape[0], Q.shape[-1]
    N = A_poly.shape[1]
    F_list = []
    for i in range(N):
        A_i = A_poly[:, i, :, :]; B_i = B_poly[:, i, :, :]
        Term = torch.bmm(A_i, Q) + torch.bmm(Q, A_i.transpose(1, 2)) + \
               torch.bmm(B_i, Y) + torch.bmm(Y.transpose(1, 2), B_i.transpose(1, 2)) + \
               2 * alpha * Q
        F_list.append(-Term)
    I = torch.eye(n, device=Q.device, dtype=Q.dtype).unsqueeze(0).expand(B_sz, -1, -1)
    F_list.append(Q - epsilon * I)
    return F_list


class LMICore(nn.Module):
    def __init__(self, n=3, m=1, N=2, alpha=0.1, epsilon=1e-5, dr_iters=100,
                 sigma=0.01, backprop="unrolling", sigma_adaptativo=False):
        super().__init__()
        self.n = n
        self.m = m
        self.N = N
        self.alpha = alpha
        self.epsilon = epsilon
        self.dr_iters = dr_iters                 # iteraciones de DR (unrolling / inferencia)
        self.sigma = sigma
        self.sigma_base = sigma                  # valor nominal (sigma puede volverse tensor)
        # sigma_adaptativo: el peso de anclaje se escala por la dispersion del politopo,
        # max_i ||A_i - A_medio||_2 (la misma metrica del reporte de la base). Los politopos
        # dispersos tienen conjunto factible mas estrecho y toleran menos anclaje al ŷ.
        self.sigma_adaptativo = sigma_adaptativo

        # y = [vec(Q simetrica), vec(Y)].  dim_Q depende solo de n; dim_Y de (m, n).
        self.dim_Q = (n * (n + 1)) // 2
        self.dim_Y = m * n
        self.dim_y = self.dim_Q + self.dim_Y
        # buffer (no parametro): asi model.to(device) lo mueve a la GPU y el
        # indexado avanzado de _vec_to_Q no cruza dispositivos (CPU idx vs CUDA tensor).
        self.register_buffer("triu_indices", torch.triu_indices(row=n, col=n, offset=0))

        # Config de la estrategia implicita (la lee ImplicitSolver)
        #   forward  : corre DR hasta el punto fijo (o max_iters) con early-stop en tol.
        self.implicit_max_iters = 4000
        self.implicit_tol = 1e-9
        #   backward : el adjunto de la IFT se resuelve MATRIX-FREE (solo VJPs), por
        #   iteracion de punto fijo estilo DEQ  w <- (J_s^T w + rhs)/(1+ridge).  El ridge
        #   regulariza la no-unicidad del punto fijo del DR (sesgo despreciable, estandar
        #   en DEQ). implicit_diag_monitor loguea el gap espectral minimo del bloque eigh
        #   (diagnostico del gradiente espectral mal condicionado; NO aborta).
        self.implicit_adjoint_iters = 1000
        self.implicit_adjoint_tol = 1e-10
        self.implicit_ridge = 1e-6
        self.implicit_diag_monitor = False
        self._set_solver(backprop)

    # ------------------- seleccion de backpropagation -----------------
    def _set_solver(self, backprop):
        if backprop in ("unrolling", "unroll", UnrollingSolver.name):
            self.solver = UnrollingSolver()
        elif backprop in ("implicit", "implicita", ImplicitSolver.name):
            self.solver = ImplicitSolver()
        else:
            raise ValueError(f"backprop desconocido: {backprop} (usa 'unrolling' | 'implicit')")
        return self

    def set_implicit(self, flag=True):
        """Compatibilidad: activa/desactiva la diferenciacion implicita."""
        return self._set_solver("implicit" if flag else "unrolling")

    @property
    def backprop(self):
        return self.solver.name

    @property
    def use_implicit(self):
        return self.solver.name == "implicit"

    @use_implicit.setter
    def use_implicit(self, flag):
        self._set_solver("implicit" if flag else "unrolling")

    def _project(self, y_hat, A_poly, B_poly):
        """Proyecta y_hat sobre el conjunto factible de la LMI usando la estrategia
        de backprop elegida (unrolling o implicita)."""
        return self.solver.project(self, y_hat, A_poly, B_poly)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("LMICore no tiene encoder; usa una subclase "
                                  "(vanilla/vertices/actuators) o llama al solver directamente.")

    # ------------------- decodificacion y -> (Q, Y) -------------------
    def _vec_to_Q(self, vec_Q):
        B_sz = vec_Q.shape[0]
        Q = torch.zeros((B_sz, self.n, self.n), device=vec_Q.device, dtype=vec_Q.dtype)
        Q[:, self.triu_indices[0], self.triu_indices[1]] = vec_Q
        Q = Q + Q.transpose(1, 2) - torch.diag_embed(Q.diagonal(dim1=-2, dim2=-1))
        return Q

    def _y_to_matrices(self, y):
        Q = self._vec_to_Q(y[:, :self.dim_Q])
        Y = y[:, self.dim_Q:].view(-1, self.m, self.n)
        return Q, Y

    # ------------------- construccion de la LMI -----------------------
    def _compute_F(self, y, A_poly, B_poly):
        # A_poly: (B, N, n, n), B_poly: (B, N, n, m)
        Q, Y = self._y_to_matrices(y)
        return lmi_blocks(Q, Y, A_poly, B_poly, self.alpha, self.epsilon)

    def _construct_L_c(self, A_poly, B_poly):
        """L (B, (N+1)n^2, dim_y) y c (B, (N+1)n^2) tal que vec(F(y)) = L y + c."""
        B_sz = A_poly.shape[0]; device = A_poly.device; dtype = A_poly.dtype
        y_zero = torch.zeros(B_sz, self.dim_y, device=device, dtype=dtype)
        c = torch.cat([F.view(B_sz, -1) for F in self._compute_F(y_zero, A_poly, B_poly)], dim=1)
        L_cols = []
        for j in range(self.dim_y):
            y_ej = torch.zeros(B_sz, self.dim_y, device=device, dtype=dtype)
            y_ej[:, j] = 1.0
            col_j = torch.cat([F.view(B_sz, -1) for F in self._compute_F(y_ej, A_poly, B_poly)], dim=1) - c
            L_cols.append(col_j.unsqueeze(-1))
        return torch.cat(L_cols, dim=-1), c

    # ------------------- forward de DR (compartido) -------------------
    def _dr_precompute(self, A_poly, B_poly):
        """Operador de la proyeccion: L, c y M_inv = (I + L^T L)^{-1}."""
        B_sz = A_poly.shape[0]
        L, c = self._construct_L_c(A_poly, B_poly)
        I_y = torch.eye(self.dim_y, device=A_poly.device, dtype=A_poly.dtype).unsqueeze(0).expand(B_sz, -1, -1)
        M_inv = torch.linalg.inv(I_y + torch.bmm(L.transpose(1, 2), L))
        # sigma adaptativo: se fija AQUI, una vez por lote, porque M_inv no depende de sigma
        # y las tres formulas que lo usan son aritmetica que difunde sobre (B,1). Fuera de
        # este modo self.sigma sigue siendo el escalar de siempre y nada cambia.
        if self.sigma_adaptativo:
            self.sigma = self._sigma_por_dispersion(A_poly)
        return L, c, M_inv

    @torch.no_grad()
    def _sigma_por_dispersion(self, A_poly):
        """sigma_i = sigma_base / (1 + dispersion_i), con dispersion = max_i ||A_i - A_medio||_2.
        Devuelve (B,1): mas dispersion -> menos anclaje al ŷ del codificador."""
        disp = torch.linalg.matrix_norm(A_poly - A_poly.mean(dim=1, keepdim=True),
                                        ord=2, dim=(-2, -1)).amax(dim=1, keepdim=True)
        return self.sigma_base / (1.0 + disp)

    def _dr_step_batch(self, y_k, x_k, y_hat, L, c, M_inv):
        """UNA iteracion de Douglas-Rachford (batcheada) sobre el estado (y_k, x_k) ->
        (y_next, x_next). Es EXACTAMENTE el cuerpo del bucle de _dr_iterate, factorizado
        para reutilizarlo (a) en _dr_iterate y (b) como la funcion de punto fijo
        T(s, y_hat) cuyo VJP usa la diferenciacion implicita matrix-free. NO altera la
        matematica del solver: mismos calculos, mismo orden que la version original."""
        B_sz = y_hat.shape[0]; block = self.n * self.n; nb = self.N + 1
        Lt = L.transpose(1, 2); s2 = 2 * self.sigma
        y_avg = (s2 * y_hat + y_k) / (s2 + 1.0)
        term2 = torch.bmm(Lt, (c - x_k).unsqueeze(-1)).squeeze(-1)
        y_w = torch.bmm(M_inv, (y_avg - term2).unsqueeze(-1)).squeeze(-1)
        x_w = torch.bmm(L, y_w.unsqueeze(-1)).squeeze(-1) + c
        xin = 2 * x_w - x_k
        xv = []
        for b in range(nb):
            X = xin[:, b * block:(b + 1) * block].view(B_sz, self.n, self.n)
            X = 0.5 * (X + X.transpose(1, 2))
            lam, V = torch.linalg.eigh(X); lam = torch.relu(lam)
            xv.append(torch.bmm(V, torch.bmm(torch.diag_embed(lam), V.transpose(1, 2))).reshape(B_sz, -1))
        x_v = torch.cat(xv, dim=1)
        x_next = x_k + (x_v - x_w)            # incr = x_v - x_w (=0 solo en el punto fijo)
        return y_w, x_next

    def _dr_state_step(self, s, y_hat, L, c, M_inv):
        """Funcion de punto fijo T(s, y_hat) en forma de estado apilado s = [y_k | x_k]
        -> [y_next | x_next]. Version (B, d) de _dr_step_single. La usa el backward
        implicito matrix-free para pedir a torch.func.vjp los productos J_s^T v y J_y^T v
        SIN materializar Jacobianas."""
        y_k = s[:, :self.dim_y]; x_k = s[:, self.dim_y:]
        y_next, x_next = self._dr_step_batch(y_k, x_k, y_hat, L, c, M_inv)
        return torch.cat([y_next, x_next], dim=1)

    def _dr_iterate(self, y_hat, L, c, M_inv, iters, tol=0.0):
        """Itera Douglas-Rachford (batcheado) y devuelve el estado final (y_k, x_k).
        Es EL MISMO forward para unrolling e implicita; la diferencia (guardar grafo
        o no) la decide quien lo llama. En el punto fijo, y_k = proyeccion Pi_C(y_hat).
        Si tol>0, para temprano cuando el residual de DR cae bajo tol."""
        y_k = y_hat.clone()
        x_k = torch.bmm(L, y_k.unsqueeze(-1)).squeeze(-1) + c          # = vec(F(y_hat))
        for it in range(iters):
            y_next, x_next = self._dr_step_batch(y_k, x_k, y_hat, L, c, M_inv)
            incr_max = (x_next - x_k).abs().max() if tol else None   # residual de DR
            y_k, x_k = y_next, x_next
            if tol and it % 25 == 0 and it > 0 and incr_max < tol:
                break
        return y_k, x_k

    def _dr_final_proj(self, y_hat, y_k, x_k, L, c, M_inv):
        """Proyeccion final sobre C1 (afin) desde el estado (y_k, x_k)."""
        y_avg = (2 * self.sigma * y_hat + y_k) / (2 * self.sigma + 1.0)
        term2 = torch.bmm(L.transpose(1, 2), (c - x_k).unsqueeze(-1)).squeeze(-1)
        return torch.bmm(M_inv, (y_avg - term2).unsqueeze(-1)).squeeze(-1)

    def _dr_step_single(self, s, y_hat, L, c, M_inv):
        """Una iteracion de DR para UN sistema (sin batch). Estado s = [y_k, x_k].
        Es la funcion de punto fijo T cuya jacobiana usa la diferenciacion implicita."""
        block = self.n * self.n; nb = self.N + 1; s2 = 2 * self.sigma
        y_k = s[:self.dim_y]; x_k = s[self.dim_y:]
        y_avg = (s2 * y_hat + y_k) / (s2 + 1.0)
        y_w = M_inv @ (y_avg - L.t() @ (c - x_k))
        x_w = L @ y_w + c
        xin = 2 * x_w - x_k
        xv = []
        for b in range(nb):
            X = xin[b * block:(b + 1) * block].view(self.n, self.n)
            X = 0.5 * (X + X.t())
            lam, V = torch.linalg.eigh(X); lam = torch.relu(lam)
            xv.append((V @ torch.diag(lam) @ V.t()).reshape(-1))
        x_v = torch.cat(xv)
        return torch.cat([y_w, x_v - x_w + x_k])
