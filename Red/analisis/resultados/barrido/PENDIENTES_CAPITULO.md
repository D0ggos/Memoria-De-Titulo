# Datos faltantes del capítulo de Experimentos (`\pendiente` de datos)

_Extraído el 2026-07-28 de los shards crudos de `analisis/resultados/barrido/`._
Convención en todo el documento: **época 400, `dr_eval=8000`, promedio sobre semillas y
sistemas** (la misma de `barrido/reporte.py`).

---

## 1. Tabla «Mejor configuración vs. receta original» — celdas que faltaban

Receta original = `paper` + `actuadores` + **implícita** + α=0.01 + n_x=3 + sz=150
(2 semillas, 10 000 filas en A2).

| | Mejor configuración | Receta original |
|---|---|---|
| Pérdida | `margen_norm` | `paper` |
| Retropropagación | Desenrollado, K_tr=30 | Implícita |
| α | 0.01 | 0.01 |
| Sistemas de entrenamiento | 400 | 150 |
| **% estabilizado (A2)** | **97.4** | **51.4** |
| **% decaimiento (A2)** | **90.9** | **43.7** |
| **iters_min mediano** | **100** | **500** |
| **κ(Q) mediano** | **1.67** | **2.31** |
| % estabilizado (A1) | 94.0 | 49.8 |

### Columna extra recomendada: aislar pérdida vs. retropropagación

La receta original cambia **dos** cosas a la vez (pérdida y gradiente). Con la misma
pérdida `paper` pero entrenada con desenrollado a K_tr=30 (α=0.01, n_x=3, sz=150):

| `paper` + ... | %estab A2 | %decay A2 | iters_min med | κ(Q) med |
|---|---|---|---|---|
| implícita (receta original) | 51.4 | 43.7 | 500 | 2.31 |
| unrolling K_tr=500 | 64.8 | 55.7 | 500 | 2.13 |
| unrolling K_tr=120 | 88.3 | 79.5 | 500 | 1.15 |
| **unrolling K_tr=30** | **94.8** | **87.6** | 250 | 3.07 |
| unrolling K_tr=5 | 94.8 | 87.3 | 250 | 8.77 |

**Lectura:** de los 46 puntos de brecha entre la receta original y la mejor configuración,
~43 los explica el cambio de retropropagación (51.4 → 94.8 con la *misma* pérdida) y solo
~2.6 la pérdida y el tamaño de datos (94.8 → 97.4).

---

## 2. Verificación de «A2 generaliza mejor que A1» — la interpretación del texto **no** se sostiene

Mejor configuración, estratificado por celda topológica (m = n_u, N = vértices):

| anillo | m | N | %estab | %decay | κ(Q) med | ‖K‖₂ med | n |
|---|---|---|---|---|---|---|---|
| A1 | 1 | 2 | 92.5 | 86.5 | 1.62 | 14.13 | 200 |
| A1 | 1 | 3 | 94.5 | 87.0 | 1.86 | 13.52 | 200 |
| A1 | 1 | 4 | 95.0 | 80.5 | 1.75 | 13.74 | 200 |
| A2 | 1 | 5 | **93.8** | 81.5 | 1.67 | 12.99 | 1000 |
| A2 | 2 | 2 | **99.1** | 95.5 | 1.73 | 25.35 | 1000 |
| A2 | 2 | 3 | **98.8** | 95.0 | 1.72 | 21.84 | 1000 |
| A2 | 2 | 4 | **98.1** | 91.7 | 1.65 | 19.81 | 1000 |
| A2 | 2 | 5 | **97.3** | 90.7 | 1.59 | 18.37 | 1000 |

El mismo patrón en la config base controlada (unrolling/dr30/α0.01/sz150/actuadores,
promedio sobre las 6 pérdidas): m=1 → 90.5–91.5 %; m=2 → 95.6–97.5 %.

**Conclusión:** el que A2 (97.4 %) supere a A1 (94.0 %) **no** se debe a que los polítopos
de mayor cardinalidad den «más vértices de información». Se debe a la **cardinalidad de
actuadores**: todas las celdas con n_u=2 rinden 97.3–99.1 % y todas las de n_u=1 rinden
92.5–95.0 %, *independientemente de N*. A2 está compuesto en 4 de sus 5 celdas por sistemas
de dos actuadores (más autoridad de control → problema intrínsecamente más fácil), y su
única celda de un actuador (N=5) rinde 93.8 %, dentro del rango de A1.
De hecho, **dentro** de n_u=2 el desempeño *baja* levemente al crecer N (99.1 → 97.3), que
es lo contrario de la hipótesis de «más vértices, más información».

Composición de los anillos (celdas presentes):
A1 = {(1,2), (1,3), (1,4)}; A2 = {(1,5), (2,2), (2,3), (2,4), (2,5)}.

---

## 3. Definición de `iters_min` — confirmada, con tres advertencias

`barrido/evaluation.py::_iters_min` (líneas 43–53): por sistema, **el menor hito de la
grilla de `dr_eval` en el que `stable = True`**, o `NaN` si nunca estabiliza. Coincide con
la definición del texto. Advertencias que conviene explicitar:

1. **Grilla discreta:** `DR_EVAL = (100, 250, 500, 1000, 2000, 4000, 8000)`
   (`barrido/config.py:28`). La mediana hereda esa cuantización (ya está en Amenazas).
2. **La mediana es condicional al éxito:** los fallos entran como `NaN` y `pandas.median`
   los descarta. `iters_min` mediano = mediana *entre los sistemas que llegaron a
   estabilizar*, no sobre la población completa.
3. **No monotonía:** la trayectoria de DR no es monótona (hay cruces transitorios), por lo
   que el código reporta siempre `iters_min` junto a `estab_en_max`; `iters_min` solo es
   concluyente cuando `estab_en_max = True`.

---

## 4. Banco de desplazamiento de polos — construcción confirmada y **una corrección**

`barrido/ood_banks.py` (líneas 9–11, 25–27, 36–42):

- **Sí es A(s) = A₀ + s·I**, con
  A₀ = [[0,1,0],[0,0,1],[2,−3,1]], B = [0,0,1]ᵀ, **N = 1 vértice**, n_x = 3.
- s ∈ linspace(−1, 2, **31** puntos). Como λ(A₀+sI) = λ(A₀)+s, s desplaza *todos* los polos
  en bloque.
- Cada sistema se normaliza dividiendo por γ = max|A,B| (escalado uniforme: no cambia el
  signo de Re λ).

### Corrección: la rampa **nunca** deja de ser factible

Verificado punto a punto con el solucionador convexo de referencia
(`analisis.benchmark._cvxpy_solve`, α = 0.01, ε = 1e−5):

| s | abscisa lazo abierto (norm.) | CVXPY factible | peor Re λ lazo cerrado | κ(Q) |
|---|---|---|---|---|
| −1.0 | −0.095 | sí | −0.147 | 1.0 |
| −0.7 | +0.005 | sí | −0.064 | 1.0 |
|  0.0 | +0.238 | sí | −0.010 | 3.4 |
| +0.5 | +0.405 | sí | −0.010 | 10.6 |
| +1.0 | +0.572 | sí | −0.010 | 42.8 |
| +1.3 | +0.672 | sí | −0.010 | 89.2 |
| +1.5 | +0.738 | sí | −0.010 | 138 |
| +2.0 | +0.905 | sí | −0.010 | 358 |

**CVXPY es factible en 31/31 puntos.** Era de esperar: con un único vértice y el par (A,B)
controlable, el sistema es siempre estabilizable. Lo que ocurre a lo largo de la rampa no
es pérdida de factibilidad sino **degradación geométrica del certificado**: κ(Q) del
certificado de referencia crece de 1 a 358 de forma casi geométrica, y el peor autovalor de
lazo cerrado se pega exactamente a −α = −0.01.

Consecuencias para el texto:

- ❌ El pie de la Tabla `tab:ood_resumen` («la rampa incluye una zona terminal donde el
  problema deja de ser factible para cualquier método, por lo que sus valores solo son
  comparables entre pérdidas») **es falso** y debe corregirse: el techo alcanzable es 100 %
  en toda la rampa, igual que en el banco del profesor. El derrumbe conjunto más allá de
  s ≈ 1.3 es un **fallo del método**, no del problema — lo que *refuerza* el argumento del
  capítulo en vez de debilitarlo.
- ❌ En Amenazas a la validez, «el banco de desplazamiento agrega del orden de un centenar
  de sistemas por punto de la rampa» **es falso**: hay **un solo sistema por punto**, el
  nominal desplazado. Los porcentajes son sobre **317 réplicas de modelo** (config ×
  semilla), no sobre sistemas. El banco del profesor sí coincide con el texto: 1 sistema
  por δ y **4 réplicas** por pérdida (2 arquitecturas × 2 semillas, 24 modelos en total).

---

## 5. Tabla `tab:ood_resumen` — los números del borrador no se reproducen

Recalculado sobre `OOD/ladder`, `dr_eval = 8000`, rama de desenrollado, promediando por
punto de rampa y luego sobre la rampa:

### Polítopo del profesor (24 modelos, 4 réplicas por pérdida) — reproduce, salvo «puntos al 100 %»

| Pérdida | media % | puntos al 100 % (borrador) | **puntos al 100 % (recalculado)** |
|---|---|---|---|
| condicionamiento | 100.0 | 100.0 | 100.0 |
| control | 100.0 | 100.0 | 100.0 |
| control_margen | 100.0 | 100.0 | 100.0 |
| paper | 100.0 | 100.0 | 100.0 |
| esfuerzo | 76.2 | 71.4 | **42.9** |
| margen_norm | 39.3 | 28.6 | **4.8** |

Las medias coinciden exactamente; la fracción de puntos resueltos al 100 % no. Curva real:
`esfuerzo` cae de 100 % a 75 % en δ = 0.9 y a 50 % en δ = 1.3 (9 de 21 puntos al 100 %);
`margen_norm` cae a 50 % ya en δ = 0.1 y a 25 % en δ = 1.1 (solo δ = 0 al 100 %).

### Desplazamiento de polos (rama de desenrollado; 44 modelos por pérdida, cobertura completa)

| Pérdida | media % | puntos al 100 % |
|---|---|---|
| **margen_norm** | **68.7** | 19.4 |
| condicionamiento | 65.5 | 32.3 |
| control_margen | 60.6 | 32.3 |
| esfuerzo | 56.2 | 25.8 |
| paper | 54.0 | 25.8 |
| control | 53.2 | 32.3 |

Se conserva la afirmación cualitativa del capítulo (`margen_norm` es la **más** robusta en
este banco, justo la que colapsa en el del profesor), pero los valores del borrador
(67.4 / 58.6 / 65.5 / 61.9 / 52.5 / 71.3) no se reproducen con ningún filtro razonable
—probablemente vienen de una versión anterior de los shards.

**Rango de `margen_norm` en deriva moderada.** El texto dice «87–93 % para s ∈ [0.1, 1.1]».
Con el marginal de desenrollado el rango real es **75.0–86.4 %**; el 85.7–100 % solo se
obtiene restringiendo además a `dr30 + α=0.01`. Hay que elegir un filtro y declararlo.

**Hueco de cobertura: cerrado el 2026-07-28.** Faltaban 7 shards OOD, todos de
`paper + unrolling + dr_train=5`. Causa: el checkpoint
`paper/unrolling/dr5/α0.001/s42` (entrenado el 11-jul 18:13) se guardó antes de que
`red/core.py` registrara el buffer `triu_indices` (editado a las 18:20), y el
`load_state_dict` estricto de `ood_eval.py` abortaba el recorrido completo al llegar a él,
dejando sin evaluar los 7 modelos que le siguen en orden alfabético. Corregido en
`barrido/ood_eval.py::load_checkpoint`: se toleran las claves ausentes que sean buffers
deterministas (reconstruidos por `build_model`) y se falla ruidosamente ante cualquier otra.
Los 348 shards están completos y la fila de `paper` ya promedia 44 modelos como el resto
(su media pasó de 53.9 a 54.0; el ordenamiento no cambia).

---

## 6. Costo de una tercera semilla

Medido en los shards `meta/` (443 corridas, 1 022 h-CPU acumuladas). `t_train_s` cubre
**solo el entrenamiento**; la evaluación de anillos se cronometra aparte y se suma abajo.

| Receta | Entrenamiento | Evaluación A1+A2 (4 checkpoints) | **Total por semilla** |
|---|---|---|---|
| Mejor config (`margen_norm`, unrolling K_tr=30, α0.01, n_x=3, **sz400**) | 0.42 h (25 min) | 16.6 min | **≈ 42 min** |
| Receta original (`paper`, **implícita**, α0.01, n_x=3, sz150) | 12.4–13.3 h | 24.0 min | **≈ 13 h** |

Ambas en paralelo ⇒ **≈ 13 h de reloj**, dominadas por la implícita. Sumar la evaluación
OOD de los dos modelos nuevos añade ~1.5 min.

Costos medianos por corrida para dimensionar otras variantes:

| backprop | K_tr | n | h/corrida (mediana) | rango | memoria pico (MB) |
|---|---|---|---|---|---|
| unrolling | 5 | 48 | 0.07 | 0.05–0.11 | 533 |
| unrolling | 30 | 240 | 0.23 | 0.05–0.58 | 618 |
| unrolling | 120 | 48 | 0.84 | 0.80–0.98 | 568 |
| unrolling | 500 | 48 | 3.50 | 2.87–4.09 | 608 |
| **implícita** | — | 59 | **13.05** | 8.82–14.13 | 541 |

Escalamientos útiles (unrolling K_tr=30, α0.01, n_x=3): sz50 → 0.09 h, sz150 → 0.23 h,
sz400 → 0.56 h (≈ lineal en el tamaño). Con n_x: 0.21 / 0.23 / 0.26 / 0.27 h para
n_x = 2/3/4/5. La rama implícita no depende de α ni de la pérdida (12.5–13.6 h en todos los
casos).

⚠️ **Todos estos tiempos se midieron con 12 corridas concurrentes en CPU.** Una corrida
sola, sin contención, será apreciablemente más rápida; tómense como cota superior.

**Recomendación.** La tercera semilla de la mejor configuración cuesta menos de una hora:
conviene correrla. La de la receta original cuesta ~13 h de reloj; si el objetivo es dar
barras de error a la comparación principal, alcanza con lanzar ambas de noche en paralelo.

---

## 7. Plataforma de cómputo y costo total del barrido

### Equipo

| Componente | Especificación |
|---|---|
| CPU | AMD Ryzen 7 5700G, 8 núcleos / 16 hilos, 3.8 GHz |
| Memoria | 32 GB DDR4-2133 (4 × 8 GB) |
| GPU | NVIDIA GeForce RTX 3050, 6 GB GDDR6 (driver 560.94, CUDA 12.4) |
| Placa base | Gigabyte B550M GAMING X WIFI6 |
| Sistema operativo | Windows 10 Home 22H2 (build 19045) |
| Entorno | Python 3.12.10, PyTorch 2.6.0+cu124, NumPy 2.5.1, SciPy 1.18.0, CVXPY 1.9.2 |

El barrido se ejecutó **íntegramente en CPU** (`device=cpu`, `float64`), con 12 procesos
concurrentes y un hilo de PyTorch por proceso (`barrido/driver.py:55,192`). La GPU quedó
sin usar: los tamaños de lote y de matriz del problema son demasiado pequeños para
amortizar la latencia de lanzamiento de núcleos, y el barrido resultó ~5× más lento en GPU
que en CPU en las mediciones preliminares.

### Costo agregado (443 corridas)

| Concepto | h-CPU | % |
|---|---|---|
| Entrenamiento | 1 022.1 | 86.6 |
| Evaluación de anillos (4 checkpoints × A1/A2 × ladder) | 158.0 | 13.4 |
| **Total** | **1 180.1** | **100** |

**1 180 h-CPU ≈ 49.2 días-CPU**, ejecutados en **≈ 4.6 días de reloj** (E1–E4: 11-jul
20:39 → 16-jul 08:02, 107.4 h; E6: 1 h 55 min el 26-jul), lo que da una aceleración
efectiva de 10.8× con 12 trabajadores.

Por etapa:

| Etapa | Corridas | h entrenamiento | h evaluación | h total |
|---|---|---|---|---|
| E1 | 240 | 846.3 | 105.4 | 951.7 |
| E2 | 48 | 11.7 | 22.6 | 34.3 |
| E3 | 36 | 10.2 | 12.8 | 23.0 |
| E4 | 47 | 141.1 | 9.7 | 150.9 |
| E6 | 72 | 12.8 | 7.4 | 20.2 |

Por rama de retropropagación:

| backprop | Corridas | h total | mediana h/corrida | % del cómputo |
|---|---|---|---|---|
| implícita | 59 (13 %) | 778.7 | 13.4 | **66.0** |
| desenrollado | 384 (87 %) | 401.4 | 0.6 | 34.0 |

La rama implícita, que es la peor en generalización, consumió **dos tercios del cómputo
total** siendo el 13 % de las corridas.

Salida cruda producida: **27 742 400 filas** de evaluación (E1 18 816 000; E2 3 091 200;
E3 2 822 400; E4 2 105 600; E6 907 200) — la cifra de «27,7 millones» del capítulo es
exacta.
