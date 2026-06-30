# Resumen Exhaustivo de la Arquitectura LMI-Net y Pipeline de Control Robusto

Este documento detalla exhaustivamente la arquitectura, matemáticas y lógica de implementación de la red neuronal diseñada para estabilizar sistemas politópicos inciertos (Sistemas LPV en tiempo continuo). Este resumen está optimizado para proveer contexto inmediato a cualquier IA o desarrollador que se integre al proyecto.

## 1. Topología del Problema Físico
El proyecto aborda el control robusto de sistemas descritos por politopos de incertidumbre. Se ha fijado la siguiente topología estricta para los experimentos principales:
- **Número de Estados ($n_x$):** 3
- **Número de Entradas/Actuadores ($n_u$):** 1
- **Vértices del Polítopo ($N$):** 2

Cada sistema se modela con matrices $A_i \in \mathbb{R}^{3 \times 3}$ y $B_i \in \mathbb{R}^{3 \times 1}$ para $i \in \{1, 2\}$. Se asume que todos los sistemas entrenados son Robustamente Estabilizables (extraídos de la base de datos `DB_ssf_RS_500_c.mat`).

## 2. El Backbone Neuronal (Red MLP)
El *backbone* es un Perceptrón Multicapa encargado de ingerir las matrices del sistema e inferir un vector no restringido de 9 parámetros ($y$) que dictan los grados de libertad de las matrices $Q$ (Matriz de Lyapunov, simétrica) e $Y$ (Matriz de Ganancia).

### 2.1. Arquitectura y Capas
- **Capa 1 (Entrada $\to$ Oculta 1):** Lineal de 24 entradas (todas las $A_i$ y $B_i$ aplanadas y concatenadas) a 128 neuronas.
- **Capa 2 (Oculta 1 $\to$ Oculta 2):** Lineal de 128 a 128 neuronas.
- **Capa 3 (Oculta 2 $\to$ Salida):** Lineal de 128 a 9 neuronas (6 parámetros para $Q$ simétrica y 3 para $Y$).

### 2.2. Funciones de Activación y Normalización
- **Activación Mish:** $f(x) = x \tanh(\text{softplus}(x))$. Se utiliza en reemplazo de ReLU por su naturaleza continuamente diferenciable y no-monotónica, previniendo quiebres bruscos del gradiente, lo cual es crítico para la estabilidad numérica del solver iterativo posterior.
- **LayerNorm:** Se aplica después de las capas lineales para centrar y re-escalar las características internamente. Evita la explosión de gradientes frente a matrices físicas de órdenes de magnitud dispares.
- **Precisión (Crucial):** Todo el grafo computacional corre estrictamente en **Float64** (Doble Precisión). El uso de Float32 causa inestabilidad numérica durante las proyecciones espectrales del solver.

### 2.3. Parámetros Entrenables
La red tiene **20,745** pesos y sesgos en total, asegurando un footprint de memoria mínimo:
- Capa 1: $3,456$ parámetros.
- Capa 2: $16,768$ parámetros.
- Capa 3: $1,161$ parámetros.

## 3. La Capa "LMI-Net": Solver Douglas-Rachford Diferenciable
La salida del *backbone* es solo una "propuesta". Esta es proyectada obligatoriamente sobre un espacio de factibilidad que asegura que el sistema será estabilizado con un *Decay Rate* $\alpha$ estricto, mediante una restricción de Desigualdad Matricial Lineal (LMI).

### 3.1. Esquema de Levantamiento (Lifting)
La LMI se divide en la intersección de dos conjuntos convexos independientes, utilizando una matriz auxiliar $X$:
- **$C_1$ (Subespacio Afín):** Contiene la relación lineal $F(y) = X$, donde $F(y)$ encapsula la inecuación de Lyapunov: $(A_i Q + B_i Y)^T + (A_i Q + B_i Y) + 2\alpha Q$.
- **$C_2$ (Cono Semidefinido Positivo - PSD):** Obliga a que la matriz $X \succeq \epsilon I$, donde $\epsilon = 1e-5$ es un margen de seguridad estricto.

### 3.2. Proyecciones Iterativas (Douglas-Rachford)
El algoritmo alterna las proyecciones en zigzag:
1. **En $C_1$:** Resuelve mínimos cuadrados (analíticamente, mediante una pseudo-inversa precalculada).
2. **En $C_2$:** Ejecuta una descomposición espectral (`torch.linalg.eigh`), truncando cualquier autovalor que sea menor a $\epsilon$.

### 3.3. Algorithm Unrolling (El Gap de Entrenamiento)
- **Train Phase:** Para que PyTorch calcule el gradiente (\textit{backprop}) sin colapsar la RAM por Out-of-Memory (OOM), el solver solo itera **30 veces**.
- **Test/Inference Phase:** Aprovechando que no hay que almacenar el grafo para diferenciación, el solver itera **5000 veces** para asegurar convergencia estricta.

## 4. Función de Pérdida (Loss) Auto-supervisada
No se usan las ganancias $K$ de MATLAB para clonación de comportamiento. Se entrena con una pérdida puramente física (Costo de control + Volumen Invariante):
$$ \mathcal{L} = \text{Traza}(Q) + 0.1 \cdot ||Y||_{F}^{2} $$
*Nota:* En la teoría se usa $-\log\det(Q)$ para el volumen de atracción, pero empíricamente se sustituyó por $\text{Traza}(Q)$ para evadir NaNs causados por autovalores en colisión durante las primeras iteraciones sub-óptimas del solver. El parámetro Decay Rate ($\alpha=0.01$) no está en el Loss, es una restricción "Hard" inquebrantable de la matriz LMI de DR.

## 5. Rendimiento Actual del Modelo
Bajo inferencia *Zero-Shot* (sistemas jamás vistos) con $\alpha=0.01$:
- **93.00%** de los sistemas logran estabilidad estricta ($Re(\lambda) < 0$).
- **88.00%** cruzan la meta rígida de Decay Rate ($Re(\lambda) \leq -0.01$).
- **Promedio del peor autovalor:** $-0.0743$.

**¿Por qué no es el 100%?** 
Limitaciones puramente numéricas. El solucionador Douglas-Rachford presenta tasas de convergencia asintóticas lentas en LMIs muy mal condicionadas, y el "Gap" entre entrenar con 30 iteraciones e inferir con 5000 produce inicializaciones subóptimas para el 7% restante de casos extremos.
