import numpy as np
import scipy.io

flag = 1  # 1: RS (Cuadráticamente Estabilizables), 2: Non-QS

if flag == 1:
    load_filename = 'DB_ssf_RS_500_c.mat'
    save_filename = 'dataset_RS.mat'
else:
    load_filename = 'DB_ssf_nonQS_100_c.mat'
    save_filename = 'dataset_nonQS.mat'

print(f"Cargando base de datos: {load_filename}...")
data = scipy.io.loadmat(load_filename)
BASE = data['BASE']
cases = int(data['cases'][0, 0])

# Preallocate lists to collect processed data
A_data_list = []
B_data_list = []
K_data_list = []
labels_nx_list = []
labels_nu_list = []
labels_N_list = []

for inputs in range(1, 3):
    if inputs == 1:
        order_ini = 2
    else:
        order_ini = 3
    for order in range(order_ini, 6):
        for vertices in range(2, 6):
            for i in range(1, cases + 1):
                elem = BASE[order - 1, inputs - 1, vertices - 1, i - 1]
                
                A_vertices = elem['A'][0, 0][0]
                B_vertices = elem['B'][0, 0][0]
                K_gain = elem['K'][0, 0]
                
                # Concatenate vertices along the 3rd dimension (axis 2 in numpy)
                A_tensor = np.dstack(A_vertices)
                B_tensor = np.dstack(B_vertices)
                
                A_data_list.append(A_tensor)
                B_data_list.append(B_tensor)
                K_data_list.append(K_gain)
                
                labels_nx_list.append(order)
                labels_nu_list.append(inputs)
                labels_N_list.append(vertices)

total_samples = len(A_data_list)

# Convert lists to NumPy object arrays (to represent MATLAB cell arrays properly)
A_data = np.empty(total_samples, dtype=object)
B_data = np.empty(total_samples, dtype=object)
K_data = np.empty(total_samples, dtype=object)

for idx in range(total_samples):
    A_data[idx] = A_data_list[idx]
    B_data[idx] = B_data_list[idx]
    K_data[idx] = K_data_list[idx]

labels_nx = np.array(labels_nx_list, dtype=np.float64)
labels_nu = np.array(labels_nu_list, dtype=np.float64)
labels_N = np.array(labels_N_list, dtype=np.float64)

print(f"Total de muestras procesadas: {total_samples}")

# Save to .mat file using scipy.io.savemat
scipy.io.savemat(
    save_filename,
    {
        'A_data': A_data,
        'B_data': B_data,
        'K_data': K_data,
        'labels_nx': labels_nx,
        'labels_nu': labels_nu,
        'labels_N': labels_N
    }
)
print(f"Base de datos guardada en: {save_filename}")