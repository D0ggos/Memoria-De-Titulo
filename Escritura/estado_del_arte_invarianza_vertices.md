# Hacia una arquitectura invariante al número de vértices para LMI-Net robusto

**Mini-documento de estado del arte y planteamiento del problema**
*Memoria de Título — extensión de LMI-Net a sistemas politópicos con cardinalidad variable*

---

## 1. Planteamiento del problema

### 1.1. Contexto: de LMI-Net a la síntesis robusta politópica

El marco base es **LMI-Net** (Tang, Goertzen & Azizan, 2026), una red neuronal con una
*capa de proyección diferenciable* que garantiza, por construcción, que la salida satisface
una desigualdad matricial lineal (LMI). Formalmente, LMI-Net aprende un mapa
parametrizado

$$\xi \;\longmapsto\; y^\*(\xi) \;\approx\; \arg\min_{y} \; c(y)
\quad \text{s.a.}\quad F(y;\xi) = F_0(\xi) + \sum_{i=1}^{m} y_i F_i(\xi) \succeq 0,$$

donde $\xi \in \mathbb{R}^p$ codifica los parámetros del sistema y la factibilidad se
impone proyectando la salida cruda de la red sobre el conjunto
$\mathcal{C} = \mathcal{C}_1 \cap \mathcal{C}_2$ (subespacio afín $\cap$ cono PSD)
mediante *Douglas–Rachford splitting*.

En el paper original, los ejemplos usan un **sistema único** bajo perturbación:
$\xi = (A, B_w)$ o $\xi = (A, B, B_w)$, aplanados a un vector fijo
(p. ej. $(\mathrm{vech}(A), \mathrm{vec}(B), \mathrm{vec}(B_w)) \in \mathbb{R}^7$).

**Nuestra extensión** aplica LMI-Net a **control robusto de sistemas politópicos**: la
planta ya no es una matriz $A$, sino un politopo de incertidumbre descrito por sus
$N$ vértices $\{(A_i, B_i)\}_{i=1}^{N}$. La condición de estabilizabilidad cuadrática
robusta con tasa de decaimiento $\alpha$ exige *una sola* función de Lyapunov común
$V(x) = x^\top P x$ ($P = Q^{-1}$) y *una sola* ganancia $K = Y Q^{-1}$ tales que

$$A_i Q + Q A_i^\top + B_i Y + Y^\top B_i^\top + 2\alpha Q \prec 0,
\qquad i = 1, \dots, N, \qquad Q \succ \epsilon I.$$

Esta extensión introduce un problema que **el paper original no aborda** (y que de hecho
lista explícitamente como trabajo futuro: *"scaling to higher-dimensional problems with
advanced backbone architectures"*): la entrada de la red pasa a ser un **conjunto de
cardinalidad variable**.

### 1.2. La asimetría estructural: entrada-conjunto, salida-certificado

El hecho central que organiza todo este documento es una asimetría entre la entrada y la
salida del mapa que queremos aprender:

| | Objeto matemático | Dimensión | Sensibilidad al orden |
|---|---|---|---|
| **Entrada** | conjunto $\{(A_i, B_i)\}_{i=1}^N$ | $\propto N$ (variable) | **invariante** (es un conjunto) |
| **Salida** | certificado $(Q, Y)$ | $n(n{+}1)/2 + m\,n$ (**fija**) | — |

La salida es **independiente de $N$** porque el método busca un certificado *común* a todo
el politopo: $Q \in \mathbb{S}^n$ e $Y \in \mathbb{R}^{m\times n}$ no crecen con el número
de vértices, sino solo con el orden $n$ y el número de actuadores $m$. Esto es la esencia
de la *estabilizabilidad cuadrática*: un único par $(Q, Y)$ válido para toda la familia
incierta.

> **Consecuencia de diseño.** El mapa a aprender es
> $\{(A_i, B_i)\}_{i=1}^N \mapsto (Q, Y)$: **conjunto de tamaño variable $\to$ vector fijo,
> invariante a permutación**. Ésa es, palabra por palabra, la definición de una función de
> conjunto en el sentido de Deep Sets. La arquitectura *debe* reflejar esta estructura.

### 1.3. Diagnóstico preciso del cuello de botella

Un análisis del código (`lmi_net.py`, `data_loader.py`) muestra que la dependencia de $N$
está **localizada en un único punto**: el encoder de entrada.

- En `data_loader.py`, las características se construyen aplanando y concatenando los
  vértices: `features = cat([A_norm.flatten(), B_norm.flatten()])`, un vector de largo
  $N\,(n^2 + n\,m)$.
- En `lmi_net.py`, la primera capa lineal queda fijada a
  `input_dim = N*(n*n) + N*(n*m)`, es decir, una matriz de pesos de forma
  $(\text{hidden}, N\,(n^2+nm))$.

Verificación empírica (red instanciada con distintos $N$):

```
N=2: input_dim=24 | W_1=(128, 24) | dim_y(salida)=9
N=3: input_dim=36 | W_1=(128, 36) | dim_y(salida)=9
N=4: input_dim=48 | W_1=(128, 48) | dim_y(salida)=9
N=5: input_dim=60 | W_1=(128, 60) | dim_y(salida)=9
```

Todo lo **posterior** ya es $N$-agnóstico:

1. La **salida** `dim_y = 9` es constante (Sección 1.2).
2. La **capa LMI / solver Douglas–Rachford** (`_compute_F`, `_construct_L_c`) itera
   `for i in range(N)` sobre `A_poly[:, i]`, generando $N{+}1$ bloques PSD; se adapta a
   cualquier $N$ que reciba, **sin pesos entrenables atados a $N$**.

> **Conclusión del diagnóstico.** Solo la primera capa lineal está acoplada a $N$.
> Reemplazar el encoder por uno de cardinalidad variable habilita una sola arquitectura
> para cualquier politopo, **conservando intactos** las capas ocultas, la cabeza de salida
> y toda la maquinaria de proyección/diferenciación implícita.

### 1.4. Por qué la concatenación plana es doblemente inadecuada

La codificación actual falla por dos razones independientes:

1. **Dimensional (rígida).** El tamaño del vector de entrada cambia con $N$, de modo que
   una red entrenada con $N=2$ ni siquiera puede *ingerir* un sistema con $N=3$.

2. **Permutacional (incorrecta como sesgo inductivo).** Aplanar
   $[\text{vértice}_1, \text{vértice}_2, \dots]$ impone un **orden arbitrario** a un objeto
   que es un conjunto: el conjunto factible de la LMI es idéntico bajo cualquier
   permutación de los vértices. Una MLP sobre el vector concatenado debe *gastar capacidad*
   aprendiendo esa invarianza a partir de datos y, aun así, no la garantiza. Un encoder
   invariante a permutación la impone *por construcción*.

---

## 2. Estado del arte: aprendizaje sobre conjuntos

El problema "función de un conjunto de cardinalidad variable, invariante a permutación" es
un área madura del aprendizaje profundo. A continuación, las familias de soluciones
relevantes, ordenadas de la más simple a la más expresiva.

### 2.1. Deep Sets y el teorema de representación

**Deep Sets** (Zaheer et al., 2017) es el resultado fundacional. Toda función invariante a
permutación sobre un conjunto admite la forma

$$f(\{x_1, \dots, x_N\}) \;=\; \rho\!\left( \textstyle\sum_{i=1}^{N} \phi(x_i) \right),$$

donde $\phi$ es una MLP compartida aplicada a *cada* elemento y $\rho$ una segunda MLP que
procesa la agregación simétrica (suma/media). La operación de agregación es independiente
de $N$ —"igual que una convolución generaliza a distintos tamaños de imagen"— y el esquema
es un **aproximador universal** de funciones invariantes a permutación. Es la solución
mínima, más interpretable y más fácil de justificar teóricamente.

### 2.2. PointNet y pooling simétrico

**PointNet** (Qi et al., 2017), originado en nubes de puntos 3D, comparte la receta de
Deep Sets pero agrega con **max-pooling**. Aporta robustez ante *outliers* y elementos
faltantes, y ofrece garantías de aproximación universal para funciones de conjunto. Útil
como variante de agregación a contrastar empíricamente (suma vs. media vs. max).

### 2.3. Redes neuronales de grafos (GNN)

Si se modela el politopo como un grafo (p. ej., los $N$ vértices como nodos de un grafo
completo), el *message passing* es equivariante a permutación y un *readout* global produce
una representación invariante. Es la opción más flexible si más adelante se desea inyectar
estructura relacional entre vértices (adyacencias, pesos baricéntricos, etc.).

### 2.4. Aplicaciones en control y sistemas dinámicos

- **Estabilización robusta de sistemas politópicos** mediante aproximaciones por red
  neuronal con certificados de estabilidad —el mismo dominio de aplicación de esta
  memoria—, que confirma la viabilidad de reemplazar solvers por redes manteniendo
  garantías.
- **Identificación de sistemas con entrada de longitud variable** usando encoders tipo
  Deep Sets, evidencia directa de que el set-encoding funciona en dinámica de sistemas.
- **Autoencoders de conjuntos con embeddings de tamaño fijo**, relevantes para producir la
  representación intermedia $N$-independiente que alimenta la cabeza MLP.

### 2.5. Conexión con optimización amortizada y *warm-start*

LMI-Net es, en esencia, un **optimizador amortizado**: aprende $\xi \mapsto y^\*(\xi)$ para
no resolver un SDP por instancia. Esta familia incluye trabajos de *warm-start* aprendido
para algoritmos de punto fijo y *splitting* (incluido Douglas–Rachford), directamente
conectados con tu capa de proyección y con la dirección de **diferenciación implícita** del
*backward*. Son la literatura puente entre "mejor encoder" (este documento) y "mejor
backward" (trabajo futuro de la memoria).

---

## 3. Mapeo al caso LMI-Net

La intervención es quirúrgica: se sustituye solo el encoder, conservando el resto.

| Componente actual | ¿Atado a $N$? | Acción |
|---|---|---|
| `features = cat([A.flatten(), B.flatten()])` | **Sí** | **Reemplazar** por encoder de conjunto |
| `nn.Linear(N·(n²+nm), 128)` (primera capa) | **Sí** | **Reemplazar** por $\phi$ compartida + pooling |
| Capas ocultas (128→128) | No | Conservar |
| Cabeza de salida (→ `dim_y`) | No | Conservar |
| Solver DR (`_compute_F`, `_construct_L_c`) | No (se adapta a $N$) | Conservar |
| Diferenciación implícita (*backward*) | No | Conservar |

**Encoder propuesto (Deep Sets como línea base).** Para cada vértice se forma el vector
$x_i = (\mathrm{vec}(A_i), \mathrm{vec}(B_i)) \in \mathbb{R}^{n^2+nm}$; una MLP compartida
$\phi$ lo lleva a un embedding; se agrega con media/max sobre los $N$ vértices (operación
invariante a permutación y a cardinalidad); y una MLP $\rho$ produce el mismo embedding de
128 dimensiones que hoy entra a las capas ocultas. El resto de la red queda **idéntico**.

**Protocolo experimental sugerido.** Entrenar con $N$ mezclados (la base
`read_database.py` ya contiene $N = 2,3,4,5$) y evaluar *generalización en $N$*: medir tasa
de estabilización y fracción de violación de la LMI para valores de $N$ vistos y no vistos,
comparando contra la red MLP actual (que requiere una arquitectura distinta por cada $N$).

---

## 4. Posicionamiento frente al paper original

- LMI-Net (2026) aprende sobre un **sistema único** con $\xi$ aplanado de tamaño fijo; no
  enfrenta cardinalidad variable y, de hecho, señala los *"advanced backbone
  architectures"* como trabajo futuro.
- La extensión politópica de esta memoria hace que la entrada sea un **conjunto** de
  vértices, exponiendo la limitación de la concatenación plana.
- Por tanto, **dotar a LMI-Net de un encoder invariante a permutación y a cardinalidad** es
  una contribución genuina y bien delimitada: aprovecha que la salida (certificado común) y
  el solver ya son $N$-independientes, y materializa exactamente el sesgo inductivo correcto
  del problema.

---

## Referencias

1. S. Tang, A. Goertzen, N. Azizan. *LMI-Net: Linear Matrix Inequality–Constrained Neural
   Networks via Differentiable Projection Layers.* arXiv:2604.05374, 2026.
2. M. Zaheer, S. Kottur, S. Ravanbakhsh, B. Póczos, R. Salakhutdinov, A. Smola. *Deep Sets.*
   NeurIPS, 2017. arXiv:1703.06114.
3. C. R. Qi, H. Su, K. Mo, L. J. Guibas. *PointNet: Deep Learning on Point Sets for 3D
   Classification and Segmentation.* CVPR, 2017. arXiv:1612.00593.
4. *Robust stabilization of polytopic systems via fast and reliable neural network-based
   approximations.* arXiv:2204.13209, 2022.
5. *Adaptive parameters identification for nonlinear dynamics using deep permutation
   invariant networks.* arXiv:2501.11350, 2025.
6. *Permutation-Invariant Set Autoencoders with Fixed-Size Embeddings for Multi-Agent
   Learning.* arXiv:2302.12826, 2023.
7. R. Sambharya, G. Hall, B. Amos, B. Stellato. *End-to-end learning to warm-start for
   real-time quadratic optimization.* L4DC, 2023. arXiv:2212.08260.
8. S. Boyd, L. El Ghaoui, E. Feron, V. Balakrishnan. *Linear Matrix Inequalities in System
   and Control Theory.* SIAM, 1994.
