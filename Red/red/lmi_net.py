import torch
import torch.nn as nn
import math

class LMINet(nn.Module):
    def __init__(self, n=3, m=1, N=2, hidden_dim=128, alpha=0.1, epsilon=1e-5, dr_iters=100, sigma=0.01):
        """
        LMI-Net: Unrolling LMIs for Robust Control.
        Para sistemas de 2 vértices (N=2), 1 actuador (m=1), orden (n=3).
        """
        super().__init__()
        self.n = n
        self.m = m
        self.N = N
        self.alpha = alpha
        self.epsilon = epsilon
        self.dr_iters = dr_iters
        self.sigma = sigma
        
        # Tamaño de entrada: N*(n*n) para A + N*(n*m) para B
        input_dim = N * (n * n) + N * (n * m)
        
        # Tamaño de y: Q es simétrica (n*(n+1)//2), Y es m*n
        self.dim_Q = (n * (n + 1)) // 2
        self.dim_Y = m * n
        self.dim_y = self.dim_Q + self.dim_Y
        
        # Backbone neuronal
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, self.dim_y)
        )
        
        # Precomputar índices de matriz simétrica
        self.triu_indices = torch.triu_indices(row=self.n, col=self.n, offset=0)

    def _vec_to_Q(self, vec_Q):
        # vec_Q shape: (B, dim_Q)
        B_sz = vec_Q.shape[0]
        Q = torch.zeros((B_sz, self.n, self.n), device=vec_Q.device, dtype=vec_Q.dtype)
        Q[:, self.triu_indices[0], self.triu_indices[1]] = vec_Q
        # Copiar a parte inferior para hacerla simétrica
        Q = Q + Q.transpose(1, 2) - torch.diag_embed(Q.diagonal(dim1=-2, dim2=-1))
        return Q

    def _y_to_matrices(self, y):
        vec_Q = y[:, :self.dim_Q]
        vec_Y = y[:, self.dim_Q:]
        Q = self._vec_to_Q(vec_Q)
        Y = vec_Y.view(-1, self.m, self.n)
        return Q, Y

    def _compute_F(self, y, A_poly, B_poly):
        # A_poly: (B, N, n, n), B_poly: (B, N, n, m)
        B_sz = y.shape[0]
        Q, Y = self._y_to_matrices(y)
        
        F_list = []
        # F^{(i)}(y) = -(A_i Q + Q A_i^T + B_i Y + Y^T B_i^T + 2*alpha*Q)
        for i in range(self.N):
            A_i = A_poly[:, i, :, :]
            B_i = B_poly[:, i, :, :]
            
            Term = torch.bmm(A_i, Q) + torch.bmm(Q, A_i.transpose(1, 2)) + \
                   torch.bmm(B_i, Y) + torch.bmm(Y.transpose(1, 2), B_i.transpose(1, 2)) + \
                   2 * self.alpha * Q
            F_list.append(-Term)
            
        # F^{(N+1)}(y) = Q - epsilon * I
        I = torch.eye(self.n, device=y.device, dtype=y.dtype).unsqueeze(0).expand(B_sz, -1, -1)
        F_list.append(Q - self.epsilon * I)
        
        return F_list # Lista de N+1 matrices (B, n, n)
        
    def _construct_L_c(self, A_poly, B_poly):
        """
        Construye explícitamente la matriz L (B, num_eq, dim_y) y c (B, num_eq)
        tal que vec(F(y)) = L * y + c.
        Como es lineal, F(y) = F(0) + sum(y_j F(e_j)).
        """
        B_sz = A_poly.shape[0]
        device = A_poly.device
        dtype = A_poly.dtype
        
        L_cols = []
        
        # c = vec(F(0))
        y_zero = torch.zeros(B_sz, self.dim_y, device=device, dtype=dtype)
        F_zero_list = self._compute_F(y_zero, A_poly, B_poly)
        c = torch.cat([F.view(B_sz, -1) for F in F_zero_list], dim=1) # (B, (N+1)*n^2)
        
        # Construir columnas de L
        for j in range(self.dim_y):
            y_ej = torch.zeros(B_sz, self.dim_y, device=device, dtype=dtype)
            y_ej[:, j] = 1.0
            
            F_ej_list = self._compute_F(y_ej, A_poly, B_poly)
            col_j = torch.cat([F.view(B_sz, -1) for F in F_ej_list], dim=1) - c # (B, (N+1)*n^2)
            L_cols.append(col_j.unsqueeze(-1))
            
        L = torch.cat(L_cols, dim=-1) # (B, (N+1)*n^2, dim_y)
        
        return L, c

    def forward(self, features, A_poly, B_poly, return_unconstrained=False):
        # 1. Salida cruda de la red neuronal
        y_hat = self.network(features)

        if return_unconstrained:
            return y_hat

        # 2. Proyectar y_hat sobre la LMI con el solver Douglas-Rachford
        return self._project_dr(y_hat, A_poly, B_poly)

    def _project_dr(self, y_hat, A_poly, B_poly):
        """
        Proyecta la propuesta y_hat sobre el conjunto factible de la LMI
        (C1 afin ∩ C2 cono PSD) mediante Douglas-Rachford splitting.
        Backbone-agnostico: solo necesita y_hat y los vertices A_poly, B_poly,
        por lo que cualquier encoder (MLP, Deep Sets, etc.) puede reutilizarlo.
        """
        B_sz = y_hat.shape[0]
        device = y_hat.device

        # 2. Precomputar L y c para la proyección
        L, c = self._construct_L_c(A_poly, B_poly)
        
        # Precomputar matriz de proyección (I + L^T L)^-1
        I_y = torch.eye(self.dim_y, device=device, dtype=y_hat.dtype).unsqueeze(0).expand(B_sz, -1, -1)
        L_T_L = torch.bmm(L.transpose(1, 2), L)
        M_inv = torch.linalg.inv(I_y + L_T_L) # (B, dim_y, dim_y)
        
        # Inicialización de DR (z = [y^T, x^T]^T). Para facilitar, se separan y, x
        y_k = y_hat.clone()
        
        # Inicializamos x con F(y_hat)
        F_y_list = self._compute_F(y_hat, A_poly, B_poly)
        x_k_list = [F.clone() for F in F_y_list]
        x_k_vec = torch.cat([X.view(B_sz, -1) for X in x_k_list], dim=1)
        
        # Bucle de Douglas-Rachford
        for k in range(self.dr_iters):
            # Paso 1: Proyección en C1 (restricción afín F(y) = X)
            # w_{k+1} = \Pi_{C1}(z_k, y_hat)
            y_avg = (1.0 / (2 * self.sigma + 1.0)) * (2 * self.sigma * y_hat + y_k)
            # y_w = (I + L^T L)^{-1} (y_avg - L^T(c - x_k_vec))
            residuo = c - x_k_vec
            term2 = torch.bmm(L.transpose(1, 2), residuo.unsqueeze(-1)).squeeze(-1)
            y_w = torch.bmm(M_inv, (y_avg - term2).unsqueeze(-1)).squeeze(-1)
            
            # x_w = L * y_w + c
            x_w_vec = torch.bmm(L, y_w.unsqueeze(-1)).squeeze(-1) + c
            
            # Reestructurar x_w en lista de matrices
            x_w_list = []
            idx = 0
            block_size = self.n * self.n
            for i in range(self.N + 1):
                x_w_list.append(x_w_vec[:, idx:idx+block_size].view(B_sz, self.n, self.n))
                idx += block_size
                
            # Paso 2: Reflexión y Proyección en C2 (cono PSD)
            # v_{k+1} = \Pi_{C2}(2 w_{k+1} - z_k)
            y_v_input = 2 * y_w - y_k # (aunque y_v no se restringe en C2, y_v = y_v_input)
            
            x_v_list = []
            for i in range(self.N + 1):
                X_in = 2 * x_w_list[i] - x_k_list[i]
                # Simetrizar para evitar errores numéricos
                X_in = 0.5 * (X_in + X_in.transpose(1, 2))
                
                # Descomposición espectral y clipping
                # En pytorch, symeig devuelve autovalores reales
                L_eig, V_eig = torch.linalg.eigh(X_in)
                L_eig = torch.relu(L_eig) # max(0, lambda)
                # Reconstruir
                X_proj = torch.bmm(V_eig, torch.bmm(torch.diag_embed(L_eig), V_eig.transpose(1, 2)))
                x_v_list.append(X_proj)
                
            x_v_vec = torch.cat([X.view(B_sz, -1) for X in x_v_list], dim=1)
                
            # Paso 3: Promediado
            y_k = y_v_input - y_w + y_k
            x_k_vec = x_v_vec - x_w_vec + x_k_vec
            
            # Reconstruir x_k_list para la siguiente iteración
            idx = 0
            for i in range(self.N + 1):
                x_k_list[i] = x_k_vec[:, idx:idx+block_size].view(B_sz, self.n, self.n)
                idx += block_size
                
        # Proyección final: \Pi_m(\Pi_{C1}(z_k, y_hat))
        y_avg = (1.0 / (2 * self.sigma + 1.0)) * (2 * self.sigma * y_hat + y_k)
        residuo = c - x_k_vec
        term2 = torch.bmm(L.transpose(1, 2), residuo.unsqueeze(-1)).squeeze(-1)
        y_star = torch.bmm(M_inv, (y_avg - term2).unsqueeze(-1)).squeeze(-1)
        
        Q_pred, Y_pred = self._y_to_matrices(y_star)
        return Q_pred, Y_pred
