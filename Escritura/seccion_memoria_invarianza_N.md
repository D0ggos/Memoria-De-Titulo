# Encoder invariante al número de vértices para LMI-Net robusto

*Sección de metodología y resultados — Memoria de Título.*
*Borrador adaptable a LaTeX. Las cifras provienen de `results_multiseed_raw.csv`
(3 semillas, 50 épocas) y `curvas_datos.txt`.*

---

## 1. Motivación y planteamiento

LMI-Net (Tang, Goertzen & Azizan, 2026) aprende un mapa parametrizado
$\xi \mapsto y^\*(\xi)$ que aproxima la solución de una optimización con restricción
LMI, imponiendo factibilidad por construcción mediante una capa de proyección
Douglas–Rachford diferenciable. En su formulación original el parámetro $\xi$ describe
un **sistema único**.

Al extender el marco a **control robusto politópico**, la planta deja de ser una matriz y
pasa a ser un politopo de incertidumbre descrito por sus $N$ vértices
$\{(A_i, B_i)\}_{i=1}^N$. Esto expone una limitación estructural del backbone MLP: su
capa de entrada tiene dimensión $N\,(n^2+nm)$, atada al número de vértices. En la práctica
ello obliga a **una arquitectura distinta por cada $N$** y, además, impone un orden
arbitrario a un objeto que es un **conjunto** (la LMI es idéntica bajo permutación de los
vértices).

**Observación clave (asimetría entrada/salida).** El certificado de salida
$y=(\mathrm{vech}(Q),\mathrm{vec}(Y))$ tiene dimensión $n(n{+}1)/2 + m\,n$,
**independiente de $N$**, porque se busca *una* función de Lyapunov común
$V(x)=x^\top Q^{-1}x$ y *una* ganancia $K=YQ^{-1}$ válidas para todo el politopo
(estabilizabilidad cuadrática). Verificamos además que el solver Douglas–Rachford ya opera
para cualquier $N$ (genera $N{+}1$ bloques sin parámetros entrenables atados a $N$). Por lo
tanto, **el único componente que impide soportar $N$ variable es el encoder de entrada.**

El mapa a aprender es entonces *conjunto de tamaño variable $\to$ vector fijo, invariante a
permutación*: exactamente una función de conjunto en el sentido de Deep Sets.

## 2. Encoder invariante propuesto

Sustituimos la concatenación plana + primera capa lineal por un **encoder Deep Sets**
(Zaheer et al., 2017), conservando intactos el resto del backbone, la cabeza de salida y la
capa de proyección:

$$x_i = [\mathrm{vec}(A_i),\,\mathrm{vec}(B_i)],\quad
h_i = \phi(x_i),\quad
z = \operatorname*{pool}_{i=1}^{N} h_i,\quad
\hat y = \rho(z),$$

donde $\phi,\rho$ son MLP compartidas y $\operatorname{pool}$ es una agregación simétrica
(media). Por construcción el encoder es invariante a permutación y a cardinalidad: una sola
arquitectura sirve para cualquier $N$. Verificamos numéricamente la invarianza (el cambio
de salida al permutar vértices es $\sim 10^{-16}$, error de máquina).

## 3. Protocolo experimental

Base de datos `DB_ssf_RS_500_c.mat`, topología $n=3$, $m=1$, con 500 sistemas robustamente
estabilizables para cada $N\in\{2,3,4,5\}$ (partición 80/20). Pérdida auto-supervisada
$\mathcal{L}=\mathrm{tr}(Q)+0.1\,\lVert Y\rVert_F^2$; solver Douglas–Rachford con 30
iteraciones en entrenamiento y 2000 en evaluación; Adam, 50 épocas. Reportamos
media$\pm$desviación sobre **3 semillas** (controlan partición e inicialización). Métricas
por $N$ sobre el conjunto de prueba: porcentaje de sistemas estrictamente estabilizados
($\mathrm{Re}(\lambda)<0$), porcentaje que cumple la tasa de decaimiento
($\mathrm{Re}(\lambda)\le-\alpha$, $\alpha=0.01$) y peor autovalor de lazo cerrado promedio.

Comparamos tres configuraciones:
- **MLP-por-N** (línea base / status quo): una arquitectura LMI-Net distinta entrenada
  para cada $N$.
- **DeepSets-mixto**: un único modelo entrenado con $N\in\{2,3,4,5\}$ mezclados.
- **DeepSets-extrap**: entrenado solo con $N\in\{2,3,4\}$ y evaluado además en $N=5$
  (cardinalidad **no vista** en entrenamiento).

## 4. Resultados

**Tabla 1. Estabilización estricta $\mathrm{Re}(\lambda)<0$ (%), media$\pm$desv. (3 semillas).**

| Método | N=2 | N=3 | N=4 | N=5 | Prom. | #redes |
|---|---|---|---|---|---|---|
| MLP-por-N (status quo) | 85±3 | 79±7 | 81±3 | 84±2 | 82.2 | 4 |
| **DeepSets-mixto** | 85±2 | **86±6** | **84±3** | **89±5** | **86.0** | **1** |
| DeepSets-extrap | 85±3 | 83±4 | 83±2 | 85±3\* | 84.0 | 1 |

\* $N=5$ no visto en entrenamiento.

**Hallazgos.**
1. **Una sola red iguala o supera a las cuatro especializadas.** DeepSets-mixto promedia
   86.0 % frente a 82.2 % del status quo, dominando o empatando en cada $N$, con una única
   arquitectura en lugar de cuatro.
2. **Generalización a cardinalidad no vista.** DeepSets-extrap alcanza 85 % en $N=5$ sin
   haberlo visto, a la par del MLP entrenado específicamente para $N=5$ (84 %). Un MLP de
   entrada fija no puede siquiera procesar un $N$ distinto al de su entrenamiento.
3. **Mayor consistencia.** La línea base presenta alta varianza en $N=3$ (79±7); el encoder
   de conjunto es más estable.

**Figura 1 (`difficulty_vs_N.png`): dificultad vs número de vértices.** A 5000 iteraciones,
la estabilización se mantiene alta en todo $N$ (85–92 %), pero el margen de estabilidad se
adelgaza monótonamente con $N$ (peor autovalor promedio $-0.093,-0.065,-0.041,-0.038$ para
$N=2,3,4,5$): a más vértices, problema más restrictivo, que el modelo único absorbe sin
colapsar.

**Figura 2 (`feasibility_vs_iters.png`): convergencia vs iteraciones.** La estabilización
crece monótonamente con las iteraciones de Douglas–Rachford y se estabiliza hacia
1000–2000. La métrica de factibilidad estricta de la LMI decrece más lentamente, reflejando
la convergencia asintótica lenta del *splitting* en LMIs mal condicionadas — propiedad del
solver (independiente del encoder) que motiva el trabajo futuro en diferenciación implícita.

## 5. Conclusión

La dependencia del número de vértices en LMI-Net robusto se elimina reemplazando únicamente
el encoder de entrada por uno invariante a permutación y a cardinalidad. La salida (un
certificado común) y la capa de proyección ya eran $N$-independientes, de modo que la
intervención es mínima y preserva todo el aparato de factibilidad-por-construcción. Los
experimentos muestran que un único modelo Deep Sets es competitivo con —o superior a— las
arquitecturas especializadas por $N$, y generaliza a cardinalidades no vistas, algo
inalcanzable para el backbone de entrada fija.
