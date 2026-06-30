import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # raíz Red/ en sys.path para correr como script

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import os

from pipeline.data_loader import RobustControlMatlabDataset
from red.lmi_net import LMINet

class RobustControlLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, Q, Y):
        # Loss combina esfuerzo de control (Frobenius norm de Y) y volumen del elipsoide (-logdet(Q))
        # Agregamos una pequeña constante en la diagonal de Q para asegurar deficiencia positiva si falla dr_iters
        epsilon = 1e-3
        I = torch.eye(Q.shape[-1], device=Q.device).unsqueeze(0).expand(Q.shape[0], -1, -1)
        # En lugar de -logdet, minimizamos la traza de Q, que es una heurística convexa común para minimizar volumen
        # y es numéricamente muy estable (evita NaNs si la matriz no es perfectamente definida positiva durante las iteraciones de DR)
        loss_vol = torch.diagonal(Q, dim1=-2, dim2=-1).sum(-1).mean()
        loss_effort = (torch.linalg.matrix_norm(Y, ord='fro', dim=(1, 2)) ** 2).mean()
        
        return loss_vol + 0.1 * loss_effort

def train_lmi_net(dataloader, epochs=50, hidden_dim=128, lr=1e-3, alpha=0.1, dr_iters=50, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LMINet(n=3, m=1, N=2, hidden_dim=hidden_dim, alpha=alpha, dr_iters=dr_iters).to(device).double()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = RobustControlLoss().to(device)
    
    if verbose:
        print(f"--- Entrenando LMI-Net en {device} con alpha={alpha} y {dr_iters} iteraciones DR ---")
        
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for features, _, A_poly, B_poly in dataloader:
            features = features.to(device)
            A_poly, B_poly = A_poly.to(device), B_poly.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass con proyecciones DR
            # Debido al gran número de iteraciones DR, PyTorch puede arrojar OutOfMemory si guardamos todo el grafo.
            # LMI-Net sugiere backprop a través del LMI layer, acá usaremos el Autograd estándar que funciona bien con ~50 iteraciones.
            Q_pred, Y_pred = model(features, A_poly, B_poly)
            
            loss = criterion(Q_pred, Y_pred)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"Época [{epoch+1}/{epochs}] | Loss: {running_loss/len(dataloader):.4f}")

    return model

def evaluate_lmi_net(model, test_loader, alpha, device, verbose=True):
    model.eval() 
    
    total_systems = 0
    stable_systems = 0
    max_eigenvalues_list = []
    
    # Aumentar iteraciones DR en inferencia para máxima factibilidad, el paper llega a usar hasta 4000-5000
    original_dr_iters = model.dr_iters
    model.dr_iters = 5000 
    
    if verbose:
        print(f"--- Evaluando LMI-Net en conjunto de Test (DR_iters={model.dr_iters}) ---")
        
    with torch.no_grad(): 
        for features, _, A_poly, B_poly in test_loader:
            features = features.to(device)
            A_poly = A_poly.to(device)
            B_poly = B_poly.to(device)
            
            Q_pred, Y_pred = model(features, A_poly, B_poly)
            
            batch_size, N, n, _ = A_poly.shape
            
            for b in range(batch_size):
                system_is_stable = True
                worst_eig = -float('inf')
                
                # Obtener la ganancia K = Y * Q^-1
                Q_inv = torch.linalg.inv(Q_pred[b])
                K_p = torch.matmul(Y_pred[b], Q_inv)
                
                for i in range(N):
                    A_i = A_poly[b, i] 
                    B_i = B_poly[b, i] 
                    
                    A_cl = A_i + torch.matmul(B_i, K_p)
                    
                    eigenvalues = torch.linalg.eigvals(A_cl)
                    max_real = torch.max(eigenvalues.real).item()
                    
                    if max_real > worst_eig:
                        worst_eig = max_real
                        
                    if max_real >= 0.0:
                        system_is_stable = False
                
                max_eigenvalues_list.append(worst_eig)
                total_systems += 1
                if system_is_stable:
                    stable_systems += 1
                    
    success_rate = (stable_systems / total_systems) * 100 if total_systems > 0 else 0
    decay_rate_success = sum(1 for eig in max_eigenvalues_list if eig <= -alpha) / total_systems * 100 if total_systems > 0 else 0
    avg_worst_eig = np.mean(max_eigenvalues_list) if max_eigenvalues_list else 0
    
    model.dr_iters = original_dr_iters # Restaurar
    
    if verbose:
        print("="*50)
        print("RESULTADOS DE LMI-NET")
        print("="*50)
        print(f"Sistemas evaluados: {total_systems}")
        print(f"Sistemas estabilizados estables (< 0): {stable_systems} ({success_rate:.2f}%)")
        print(f"Sistemas que cumplen Decay Rate <= {-alpha}: {decay_rate_success:.2f}%")
        print(f"Promedio del Peor Autovalor: {avg_worst_eig:.4f}")
        print("="*50)
        
    return success_rate, avg_worst_eig

def main():
    MAT_FILE = 'DB_ssf_RS_500_c.mat'
    
    if not os.path.exists(MAT_FILE):
        print(f"Error: {MAT_FILE} no encontrado.")
        return
        
    # Cargamos el dataset con las características pedidas
    full_dataset = RobustControlMatlabDataset(MAT_FILE, order=3, inputs=1, vertices=2)
    
    total_size = len(full_dataset)
    if total_size == 0:
        print("Error: Sin muestras para N=2, m=1, n=3.")
        return
        
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    alpha_decay = 0.01
    
    print(f"Total de sistemas en BD para esta topología: {total_size}")
    modelo = train_lmi_net(train_loader, epochs=50, hidden_dim=128, alpha=alpha_decay, dr_iters=30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_lmi_net(modelo, test_loader, alpha=alpha_decay, device=device)

if __name__ == "__main__":
    main()
