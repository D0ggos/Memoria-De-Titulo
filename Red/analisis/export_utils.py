import pandas as pd

def export_results_to_excel(results_list, filename="resultados_cv.xlsx"):
    """
    Exporta los resultados de Cross Validation a un archivo Excel.
    Resalta en verde la fila con el mejor resultado.
    
    Args:
        results_list (list): Lista de diccionarios. Cada diccionario debe tener 
                             la forma: {'params': {...}, 'mean_success': float, 'mean_eig': float}
        filename (str): Nombre del archivo de salida.
    """
    # 1. Preparar los datos (aplanar el diccionario de parámetros)
    data_for_df = []
    for entry in results_list:
        row = entry['params'].copy()
        row['Success_Rate'] = entry['mean_success']
        row['Mean_Eig'] = entry['mean_eig']
        data_for_df.append(row)
    
    df = pd.DataFrame(data_for_df)

    # 2. Definir la lógica para resaltar el mejor
    # El mejor es el de mayor Success_Rate, y en caso de empate, menor Mean_Eig
    idx_best = df.sort_values(by=['Success_Rate', 'Mean_Eig'], ascending=[False, True]).index[0]

    def highlight_best(row):
        return ['background-color: #c6efce; color: #006100' if row.name == idx_best else '' for _ in row]

    # 3. Aplicar estilo y exportar
    styled_df = df.style.apply(highlight_best, axis=1)
    
    try:
        styled_df.to_excel(filename, index=False, engine='openpyxl')
        print(f"\n[OK] Resultados exportados exitosamente a: {filename}")
    except Exception as e:
        print(f"\n[Error] No se pudo exportar el Excel: {e}")

if __name__ == "__main__":
    # Ejemplo de uso/test
    test_results = [
        {'params': {'lr': 0.01, 'hidden': 64}, 'mean_success': 85.5, 'mean_eig': 0.12},
        {'params': {'lr': 0.001, 'hidden': 128}, 'mean_success': 92.0, 'mean_eig': 0.05},
        {'params': {'lr': 0.1, 'hidden': 32}, 'mean_success': 70.2, 'mean_eig': 0.45},
    ]
    export_results_to_excel(test_results)
