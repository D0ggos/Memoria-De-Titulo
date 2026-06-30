import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np

class RobustControlMatlabDataset(Dataset):
    def __init__(self, mat_filepath, order=3, inputs=1, vertices=2):
        """
        Ingesta la base de datos de MATLAB y filtra por topología de sistema.
        
        Parámetros:
        - mat_filepath: Ruta al archivo .mat (ej. 'DB_ssf_RS_500_c.mat')
        - order (n): Dimensión del estado x(t).
        - inputs (m): Dimensión de la entrada u(t).
        - vertices (N): Cantidad de vértices del politopo.
        """
        super().__init__()
        # Guardamos hiperparámetros para control de dimensiones
        self.num_order = order
        self.num_inputs = inputs
        self.num_vertices = vertices

        print(f"Cargando base de datos desde {mat_filepath}...")
        mat_data = sio.loadmat(mat_filepath, struct_as_record=False, squeeze_me=True)
        
        # Extraemos el Cell Array principal
        base_cell = mat_data['BASE']
        
        # Mapeo de índices MATLAB (1-based) a Python (0-based)
        idx_order = order - 1
        idx_inputs = inputs - 1
        idx_vertices = vertices - 1
        
        # Extraemos todos los casos ('i') para esta configuración específica
        raw_cases = base_cell[idx_order, idx_inputs, idx_vertices, :]
        
        # Filtramos posibles celdas vacías
        self.cases = [c for c in raw_cases if not isinstance(c, (float, int)) and hasattr(c, 'A')]
        self.num_cases = len(self.cases)
        
        print(f"Dataset inicializado: {self.num_cases} sistemas válidos encontrados para (n={order}, m={inputs}, N={vertices}).")

    def __len__(self):
        return self.num_cases

    def __getitem__(self, idx):
        matlab_struct = self.cases[idx]
        
        # A y B son cell arrays de matrices (los vértices)
        A_verts = np.stack(matlab_struct.A)  # (N, n, n)
        B_verts = np.stack(matlab_struct.B)  # (N, n, m)
        K_target = matlab_struct.K           # (m, n)
        
        # Conversión a tensores y asegurar dimensiones (usando double precision para estabilidad numérica)
        A_tensor = torch.tensor(A_verts, dtype=torch.float64)
        B_tensor = torch.tensor(B_verts, dtype=torch.float64)
        K_tensor = torch.tensor(K_target, dtype=torch.float64)

        # Forzar dimensiones correctas si fueron comprimidas por squeeze_me
        if A_tensor.dim() == 2: # Caso N=1
            A_tensor = A_tensor.unsqueeze(0)
        if B_tensor.dim() == 2: # Caso m=1 o N=1
            # Si m=1 y N>1, B_verts es (N, n). Queremos (N, n, 1)
            # Si N=1 y m>1, B_verts es (n, m). Queremos (1, n, m)
            # Para ser robustos, comparamos con los parámetros n, m, N
            if B_tensor.shape == (self.num_vertices, self.num_order):
                B_tensor = B_tensor.unsqueeze(-1)
            elif B_tensor.shape == (self.num_order, self.num_inputs):
                B_tensor = B_tensor.unsqueeze(0)
        
        if K_tensor.dim() == 1: # Caso m=1
            K_tensor = K_tensor.unsqueeze(0)
        
        # --- NUEVO: NORMALIZACIÓN FÍSICA ---
        # 1. Encontramos el valor máximo absoluto en todo el politopo (A y B)
        max_A = torch.max(torch.abs(A_tensor))
        max_B = torch.max(torch.abs(B_tensor))
        gamma = torch.max(max_A, max_B)
        
        # 2. Prevenimos división por cero (por si hay un sistema nulo)
        if gamma == 0:
            gamma = 1.0
            
        # 3. Escalamos A y B. 
        A_norm = A_tensor / gamma
        B_norm = B_tensor / gamma
        
        # Feature vector usando las matrices normalizadas
        features = torch.cat([A_norm.flatten(), B_norm.flatten()])
        
        # Devolvemos los tensores. Para el cálculo de la Loss Física (M_i), 
        # usaremos A_norm y B_norm para mantener la estabilidad numérica.
        return features, K_tensor, A_norm, B_norm
