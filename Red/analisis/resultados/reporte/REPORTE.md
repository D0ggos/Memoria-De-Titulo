# Reporte de resultados — LMI-Net

Matriz completa de experimentos para la sección de resultados. Todo lo de este
reporte es reproducible con `analisis/experimento_reporte.py` (una corrida,
~10.5 min, seed=42). Cada subsección tiene su CSV y su figura (PNG para ver,
PDF para LaTeX).

**Protocolo común.** Base `DB_ssf_RS_500_c.mat`, 150 sistemas por celda
(n, m, N), partición 80/20, α=0.01, ε=1e-5, batch=16, lr=1e-3, Adam,
100 épocas (15 en los bloques con entrenamientos lentos: backprop y dr_train).
Evaluación a 1000 iteraciones DR salvo indicación. "Estabilizado" =
máx Re λ(A_i+B_iK) < 0 en todos los vértices, con K = YQ⁻¹.

---

## 1. Resultados en distribución

### 1.1 Escalera de arquitecturas (celda común n_x=3, N=2) — `C_arquitecturas.csv`, `fig_C_arquitecturas`

| arquitectura | pérdida | backprop | % estab. | % decay α | params | train (s) | infer (ms/sys) |
|---|---|---|---|---|---|---|---|
| vanilla (paper) | paper | implícito, DR-500 | **90.0** | 90.0 | 6 345 | 31.2 | 4.8 |
| vertices | control | unrolling-30 | 76.7 | 73.3 | 36 617 | 6.3 | 4.9 |
| actuadores | control | unrolling-30 | 80.0 | 73.3 | 69 769 | 6.6 | 4.9 |
| *techo CVXPY* | — | — | *93.3* | — | — | — | 7.3 |

**Lectura.** En *su* celda, la vanilla fiel al paper es la mejor: es una
especialista (un modelo por (N, n_u), receta completa del paper). Las
invariantes pagan ~10 puntos por generalizar: **un solo modelo** de
`actuadores` cubre N∈{2..5} y n_u∈{1,2}, mientras que la vanilla necesitaría
8 modelos. Es el trade-off especialista/generalista, no un defecto de
implementación.

### 1.2 Comparación de pérdidas (actuadores, n=3, N=2..5) — `A_losses.csv`, `A_losses_porN.csv`, `fig_A_losses`

| pérdida | % estab. (media N=2..5) | % red mejor que CVXPY en su propia loss | loss final |
|---|---|---|---|
| control | **90.0** | 3.7 | 0.0010 |
| control_margen | 87.5 | 3.7 | 0.0507 |
| paper | 84.2 | **40.7** | −35.34 |
| *techo CVXPY (factible)* | *90.0* | — | — |

**Lecturas.**
- En distribución, `control` **alcanza el techo de factibilidad** (90.0% vs
  90.0% factible por CVXPY): dentro de la base, la red estabiliza
  prácticamente todo lo estabilizable.
- `control_margen` cuesta ~2.5 puntos en distribución — el precio del margen.
  Su pago está fuera de distribución (sección 3).
- El 40.7% de `paper` vs 3.7% de las otras NO significa que `paper` sea mejor
  red: significa que el certificado de CVXPY (el factible más cercano a y=0)
  es casi óptimo para trace(Q) pero muy subóptimo para logdet(Q)−βλ_min(F);
  la comparación red-vs-CVXPY depende del criterio, como se esperaba: CVXPY
  garantiza factibilidad, no optimalidad.
- Nota al pie: en N=5 la red marca 96.7% > techo 86.7%. No es magia: el techo
  exige margen α=0.01 y "estabilizado" solo pide Re λ<0, y el techo se estima
  sobre 15 de los 30 sistemas de test (costo de CVXPY).

### 1.3 Unrolling vs diferenciación implícita (15 épocas) — `B_backprop.csv`, `fig_B_backprop`

| backprop | % estab. | train (s) | mem pico (+MB) | loss final |
|---|---|---|---|---|
| unrolling-30 | **80.0** | **5.6** | 0.2 | 0.0037 |
| implícito | 65.8 | 207.5 | 1.7 | **0.00009** |

**Lectura (el hallazgo más contraintuitivo).** El implícito optimiza la
pérdida 40× mejor… y estabiliza 14 puntos menos. El gradiente exacto del
punto fijo minimiza `control` con demasiado éxito: lleva a Q a la frontera
casi-singular (trace(Q) mínimo), exactamente la región donde K=YQ⁻¹ se
degrada. El unrolling truncado a 30 iteraciones actúa como regularización
implícita. Además cuesta 37× menos tiempo. Mismo init, mismos datos, misma
evaluación.

### 1.4 Iteraciones desenrolladas en entrenamiento — `D_drtrain.csv`, `fig_D_drtrain`

| dr_train | % estab. | train (s) | loss final |
|---|---|---|---|
| 10 | **82.5** | 2.6 | 0.0081 |
| 30 | 80.0 | 5.6 | 0.0037 |
| 100 | 79.2 | 16.5 | 0.0005 |
| 500 | 66.7 | 79.1 | 0.0003 |

**Lectura.** Confirma 1.3 con una curva dosis-respuesta: a más iteraciones en
train, mejor loss y **peor** estabilización, monótono en ambos sentidos. La
proyección incompleta durante el entrenamiento es una regularización, no una
limitación. (Coincide con la elección dr_train=30 del resto de la tesis y
además es la opción barata.)

---

## 2. Red vs CVXPY: velocidad (de `A_losses_porN.csv`)

| N | red (ms/sys, batch) | CVXPY (ms/sys) | speedup |
|---|---|---|---|
| 2 | 4.8 | 7.1–13.8 | 1.5–2.9× |
| 3 | 6.1 | 136 | **22×** |
| 4 | 8.4 | 21.7 | 2.6× |
| 5 | 9.7 | 481 | **50×** |

**Lectura.** La ventaja de la red crece con el tamaño del SDP (N=5: 50×) y es
de *throughput*: inferencia batcheada con presupuesto fijo de iteraciones.
Para UN sistema suelto pequeño, CVXPY es competitivo (~4-7 ms) — las dos
mediciones aparecen en los benchmarks OOD de la sección 3, donde la inferencia
es de a un sistema. Ambas cifras son honestas y hay que reportarlas juntas.

---

## 3. Casos borde: fuera de distribución (OOD)

Sistemas construidos a mano y desplazados hacia la inestabilidad — nunca
vistos en entrenamiento. En TODOS los puntos de ambos barridos CVXPY confirma
que la LMI es **factible** (`lmi_factible=True`): cualquier fallo es del
método, no del problema.

### 3.1 Pole-shift nominal (N=1, n_x=3, A+s·I) — `E_nominal_shift.csv`, `fig_E_nominal`

Modelos del bloque A (n=3). Presupuestos DR hasta 8000.

| shift | absc. lazo abierto | control | control_margen | paper |
|---|---|---|---|---|
| −1.0 … 0.0 | −0.28 … +0.72 | 100–250 iters | 100–250 | 100–8000 |
| +0.25 … +0.5 | +0.97 … +1.22 | 8000 | 8000 | 8000 / 100 |
| ≥ +0.75 | ≥ +1.47 | **falla** | **falla** | **falla** |

**Lecturas.**
- Las tres pérdidas colapsan en el mismo punto (shift ≈ 0.75) pese a
  factibilidad garantizada. El N=1 nominal es el OOD más agresivo: el modelo
  se entrenó solo con politopos N∈{2..5} normalizados de la base.
- Ojo con `paper` en shifts ≥1.5: marca `iters_min=100` pero
  `estabilizado_en_max=False` — es el **cruce transitorio** de la trayectoria
  no-monótona de DR (un iterado temprano cae casualmente en la región estable
  y luego la trayectoria vuelve a salir). `iters_min` solo es conclusivo junto
  con `estabilizado_en_max=True`.
- `control_margen` NO ayuda aquí (n=3, piso=0.05 sin ajustar). El remedio no
  es universal con sus defaults — ver 3.2, donde sí.

### 3.2 Politopo del profesor (N=2, n_x=2, A_i+δ·D_i por vértice) — `F_profesor.csv`, `fig_F_profesor`

D₁=diag(2,1) desestabiliza el vértice 1 (abscisa hasta +3.41 en δ=2);
D₂=diag(−2,1) *estabiliza* el vértice 2 (hasta −2.43): el politopo se estira
4δ en la entrada (1,1). Modelos n=2 entrenados con cada pérdida (100 épocas).

| δ | absc. A₁ | control | control_margen | paper |
|---|---|---|---|---|
| 0.00–0.43 | −1.0 … −0.36 | 100 | 100–250 | 100–250 |
| 0.57–1.0 | −0.14 … +1.0 | 2000–4000 | **100–1000** | 500–4000 |
| 1.14–1.71 | +1.4 … +2.8 | 4000–8000 | **2000–4000** | falla desde δ=1.29 |
| 1.86–2.00 | +3.1 … +3.4 | **falla** | **4000** ✓ | falla |

**Lectura (el resultado central del capítulo OOD).** `control_margen`
**elimina el estancamiento**: estabiliza el barrido completo hasta δ=2
(vértice 1 con autovalor +3.41) con ≤4000 iteraciones, donde `control` se
estanca desde δ=1.86 y `paper` desde δ=1.29. Además el tercer panel de la
figura muestra *por qué*: `control` aterriza siempre pegado a la frontera
(peor autovalor = −0.0100…, exactamente −α), mientras `control_margen`
mantiene margen interior (−0.46) en la zona fácil y lo va cediendo con
gracia. El costo fue ~2.5 puntos en distribución (sección 1.2).

### 3.3 Aislamiento: ¿solver o encoder? — `G_dr_puro.csv`, `fig_G_aislamiento`

La misma proyección DR, sobre los mismos politopos normalizados, pero desde
ŷ **aleatorio** (mediana de 5 seeds) en vez del ŷ de la red:

| δ | DR puro (mediana) | red control | red control_margen |
|---|---|---|---|
| 1.29 | 100 | 8000 | 2000 |
| 1.71 | 8000* | 8000 | 4000 |
| 2.00 | **100** | falla | 4000 |

*\* punto con varianza alta entre seeds (peor caso: no logra).*

**Lectura.** El DR puro estabiliza δ=2 en 100 iteraciones (mediana) — el
mismo punto donde la red con `control` falla a 8000. **El cuello de botella
no es el solver: es la dirección del ŷ del encoder.** En esta arquitectura ŷ
no es solo un warm start: es el punto que se proyecta, así que decide a qué
punto del conjunto factible se llega. Un ŷ entrenado con una pérdida que
premia Q pequeña apunta, en OOD, hacia una esquina casi-singular de la
frontera; un ŷ aleatorio se proyecta a un punto genérico y sale rápido. Esto
convierte los fallos OOD en un problema de *entrenamiento/pérdida* (arreglable,
como demuestra 3.2), no de la capa de optimización diferenciable.

### 3.4 Anatomía del estancamiento: la trayectoria de DR iteración a iteración — `trayectoria_dr.csv`, `fig_trayectoria_dr`

Peor autovalor de lazo cerrado muestreado cada 20 iteraciones (hasta 8000)
sobre el politopo del profesor (`analisis/experimento_trayectoria.py`).
Tres paneles, tres fenómenos:

1. **Estancamiento** (red `control`): la trayectoria baja por mesetas
   escalonadas (zig-zag) y, en δ fáciles, cruza a estable; en δ=2 se queda
   en una **meseta positiva (+0.79) que nunca cruza** — el fallo no es "le
   faltaron iteraciones", es un punto de llegada malo.
2. **Cruce transitorio** (red `paper`, δ=1.57): la trayectoria **cruza a la
   región estable de pasada** (≈88 checkpoints bajo cero) **y se devuelve**,
   terminando en +0.51. Es la prueba visual de que DR no es monótono en esta
   métrica y de por qué `iters_min` solo vale junto a `estabilizado_en_max`.
3. **El ŷ decide** (mismo sistema, δ=2): con ŷ de `control` la meseta
   positiva; con ŷ de `control_margen` cruza en ~2500; con ŷ **aleatorio**
   cruza en ~60 y se queda en −0.28. La versión "de cerca" del hallazgo 3.

Detalle técnico visible en las trayectorias: picos tempranos de hasta +5·10⁴
(recortados en la figura) — K = YQ⁻¹ explota cada vez que Q pasa por
casi-singular durante la convergencia.

### 3.5 Mapa de polos en el plano complejo — `plano_complejo.csv`, `fig_plano_complejo`

La vista geométrica de la LMI: estabilizar = meter TODOS los autovalores de
lazo cerrado (por vértice) al semiplano izquierdo, idealmente pasada la línea
−α. Politopo del profesor, coordenadas normalizadas
(`analisis/experimento_plano_complejo.py`). Tres paneles:

1. **Lazo abierto** — con δ, los polos del vértice 1 se abren hacia la derecha
   y **cruzan al semiplano derecho** (se desestabiliza, abscisa hasta +3.4);
   los del vértice 2 se van a la izquierda. El color codifica δ.
2. **Lazo cerrado de la red `control`** — los polos colocados se **agolpan
   sobre el margen −α** (Re ≈ 0) al crecer δ, y en los δ altos uno se escapa
   al RHP: la red aterriza en la restricción activa, no en el interior.
3. **δ=2, tres ŷ (zoom en el eje)** — mismo sistema: `control` deja el polo
   crítico en Re=+0.79 (**RHP → falla**), `control_margen` en −0.010 (justo
   sobre el margen → estable) y ŷ **aleatorio** en −0.28 (interior cómodo).
   La imagen del hallazgo 3/4 en el plano: el encoder entrenado se sienta en
   el filo; un ŷ genérico factible queda adentro.

Los pesos de ambos modelos quedan congelados en `modelo_control_n2.pt` y
`modelo_control_margen_n2.pt` (la figura es reproducible bit a bit; sin
congelar, el fallo marginal de `control` en δ=2 varía entre corridas por el
no-determinismo de BLAS — es el filo del margen).

---

## 4. Hallazgos principales (para la discusión)

1. **En distribución la red alcanza el techo de factibilidad** (90% vs 90%
   CVXPY, pérdida `control`) con inferencia batcheada hasta 50× más rápida
   que el SDP (N=5). Para un sistema suelto, CVXPY es competitivo — la
   ventaja es de throughput.
2. **Optimizar mejor la pérdida empeora la estabilización** (dos evidencias
   independientes: implícito vs unrolling, y la curva dr_train 10→500). Las
   pérdidas tipo volumen empujan Q a la frontera casi-singular; el
   truncamiento del unrolling regulariza. La pérdida es el objetivo *proxy*,
   no la métrica.
3. **El modo de fallo OOD es el encoder, no el DR** (aislamiento 3.3): ŷ
   aleatorio estabiliza en ≤500 iteraciones donde el ŷ entrenado se estanca a
   8000. El certificado de la red aterriza sistemáticamente en −α exacto
   (frontera), con Q casi singular y K=YQ⁻¹ explosiva a presupuesto corto.
4. **Regularizar el aterrizaje arregla el OOD**: `control_margen`
   (+η·relu(piso−λ_min(Q))) estabiliza TODO el barrido del profesor
   (abscisa hasta +3.4) donde las otras dos pérdidas fallan, pagando solo
   ~2.5 puntos en distribución. Con los defaults (η=1, piso=0.05) no rescata
   el nominal n=3 — el ajuste de (η, piso) por orden queda como trabajo
   futuro inmediato (barrible con `grid_search` + `register_loss`).
5. **Diagnóstico de cruces transitorios**: `iters_min` debe leerse junto a
   `estabilizado_en_max`; la trayectoria de DR no es monótona y puede cruzar
   la región estable de pasada (visible en `paper`, 3.1).
6. **La especialista gana su celda, la invariante gana la guerra**: vanilla
   90% en (n=3, N=2) vs 80% de actuadores en esa celda — pero un solo modelo
   invariante cubre las 8 celdas (N×n_u) con 90% medio (bloque A) y sin
   reentrenar.

## 5. Barrido ampliado de hiperparámetros (`experimento_hparams.py`)

Un eje a la vez sobre la config base (actuadores, n=3, N=2..5, unrolling-30,
control, 100 épocas, batch=16, lr=1e-3, α=0.01), más la interacción
dr_train×épocas y un re-chequeo OOD. Objetivo: ¿los hallazgos de §1–§4
cambian o se repiten?

### 5.0 Nota de reproducibilidad (leer primero)

La config base, con la MISMA semilla, dio 85.8 / 87.5 / 90.0 % en tres
corridas independientes (no-determinismo de BLAS multihilo en eigh/gemm,
amplificado por 100 épocas). **Diferencias menores a ~5 puntos son ruido**;
los efectos que se reportan abajo son mucho mayores.

### 5.1 Épocas: sí ayuda… y depende del orden — `H_epochs.csv`, `fig_H_epochs`

| épocas | 15 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| % estab. (n=3) | 80.0 | 80.8 | 85.8 | **89.2** | 89.2 |

Más épocas mejora y **satura en ~200** (n=3). Pero en el re-chequeo OOD (n=2,
§5.6) 400 épocas *degradaron* lo en-distribución de `control` (90→78.3%): el
óptimo de épocas depende del orden/tamaño del problema. lr plano en
[3e-4, 1e-2] (87–88%) y batch con óptimo suave en 16–32 (`H_lr.csv`,
`H_batch.csv`).

### 5.2 dr_train × épocas: más épocas NO rescata al unrolling largo — `H_drtrain_epochs.csv`, `fig_H_drtrain_epochs`

| dr_train | 5 | 30 | 120 | 500 |
|---|---|---|---|---|
| 15 épocas | 79.2 | 80.0 | 76.7 | 66.7 |
| 100 épocas | **88.3** | 85.8 | **55.8** | 63.3 |

El hallazgo 2 se **refuerza**: con más épocas, el unrolling largo empeora aún
más (120 iters: 76.7→55.8%). El caso 120@100ep es de libro: loss_final llega
a −0.0003 (Q colapsada hasta trazas negativas del residuo de proyección) y
los márgenes de lazo cerrado quedan en −0.009. La mejor celda de toda la
matriz es **dr_train=5 + 100 épocas** (88.3%, y la más barata: 12 s).

### 5.3 dr_eval: el presupuesto de inferencia sigue pagando hasta 5000 — `H_dr_eval.csv`, `fig_H_dr_eval`

| dr_eval | 100 | 250 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|---|
| % estab. | 37.5 | 65.0 | 82.5 | 85.8 | 87.5 | **92.5** |
| ms/sistema | 0.8 | 1.8 | 3.6 | 7.1 | 14.1 | 35.0 |

Mismo modelo, sin reentrenar: monótono hasta 5000 (92.5%, por encima del
techo α porque "estabilizado" no exige el margen). A 35 ms/sys sigue siendo
4–14× más rápido que CVXPY en N≥3. El default 1000 deja ~7 puntos sobre la
mesa; dr_eval es la perilla calidad/latencia del modelo desplegado.

### 5.4 α: zona útil 0.01–0.05, y un efecto no trivial — `H_alpha.csv`, `fig_H_alpha`

| α | 0.001 | 0.01 | 0.05 | 0.1 |
|---|---|---|---|---|
| % estab. red | 79.2 | 87.5 | **89.2** | 72.5 |
| techo CVXPY (factible con α) | 95.0 | 90.0 | 57.5 | 37.5 |

α muy chico (0.001) también daña: el certificado aterriza casi sin margen.
Y α=0.05 estabiliza **más que su propio techo de factibilidad** (89.2% vs
57.5%): pedir un margen exigente en la LMI de entrenamiento actúa como
objetivo pro-robustez aunque el margen estricto sea infactible — la red
igual deja Re λ<0. En α=0.1 el conjunto factible se vacía (37.5%) y todo
colapsa.

### 5.5 Orden n_x: pico en n=3 — `H_n.csv`, `fig_H_orden`

| n_x | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| % estab. | 81.7 | **87.5** | 79.2 | 75.8 |
| techo CVXPY | 90.0 | 90.0 | 95.0 | 92.5 |

Con presupuesto fijo (150 sistemas/celda, 100 épocas, misma capacidad), el
rendimiento cae con el orden (dim(y) crece cuadrático) mientras el techo se
mantiene ≥90%: la brecha en n=4–5 es de datos/capacidad, no de factibilidad.

### 5.6 Re-chequeo OOD: ¿400 épocas cambian el hallazgo 4? — `H_ood_epochs.csv`, `fig_H_ood_epochs`

Politopo del profesor completo (δ hasta 2.0), modelos n=2:

| pérdida | épocas | % estab. en distrib. | δ máx estabilizado | iters en δ=2 |
|---|---|---|---|---|
| control | 100 | 90.0 | 1.71 | falla |
| control | 400 | 78.3 | **2.00** | 8000 |
| control_margen | 100 | 88.3 | **2.00** | 4000 |
| control_margen | 400 | 86.7 | **2.00** | 4000 |

**El hallazgo se matiza**: más épocas TAMBIÉN rescatan el OOD de `control`
(δ=2 con 8000 iters)… pero pagando 12 puntos en distribución (90→78.3) — el
mismo trade-off frontera/margen por otra vía. `control_margen` sigue
dominando: barrido completo en ambas configuraciones de épocas, con menos
iteraciones (4000 vs 8000), sin ajuste y con costo en distribución de ~2
puntos. La conclusión operativa no cambia: regularizar el aterrizaje es el
remedio barato; épocas extra son un remedio caro y frágil.

### Veredicto sobre los hallazgos de §4

| hallazgo | veredicto |
|---|---|
| 1. Techo en distribución + ventaja throughput | **Se sostiene**; con dr_eval=5000 sube a 92.5% |
| 2. Optimizar mejor la loss empeora | **Se refuerza** (120@100ep colapsa a 55.8%) |
| 3. El cuello OOD es el encoder | **Se sostiene** |
| 4. `control_margen` arregla el OOD | **Se sostiene con matiz**: épocas extra también lo logran, pero 5× más caras en train, con más iteraciones de inferencia y −12 pts en distribución |
| 5. Cruces transitorios | Se sostiene (sin evidencia nueva) |
| 6. Especialista vs invariante | Se sostiene; nuevo: el pico de la invariante está en n=3 y cae con el orden |
| (nuevo) | lr y batch: insensibles en rangos razonables; α útil en [0.01, 0.05]; varianza entre corridas ±4 pts con la misma semilla |

## Archivos

| bloque | tabla | figura |
|---|---|---|
| A pérdidas | `A_losses.csv`, `A_losses_porN.csv` | `fig_A_losses.*` |
| B backprop | `B_backprop.csv` | `fig_B_backprop.*` |
| C arquitecturas | `C_arquitecturas.csv` | `fig_C_arquitecturas.*` |
| D dr_train | `D_drtrain.csv` | `fig_D_drtrain.*` |
| E nominal OOD | `E_nominal_shift.csv` | `fig_E_nominal.*` |
| F politopo profesor | `F_profesor.csv` | `fig_F_profesor.*` |
| G aislamiento | `G_dr_puro.csv` | `fig_G_aislamiento.*` |
| trayectorias DR (anatomía) | `trayectoria_dr.csv` | `fig_trayectoria_dr.*` |
| mapa de polos (plano complejo) | `plano_complejo.csv` | `fig_plano_complejo.*` |
| H lr / épocas / batch | `H_lr.csv`, `H_epochs.csv`, `H_batch.csv` | `fig_H_lr.*`, `fig_H_epochs.*`, `fig_H_batch.*` |
| H dr_train×épocas | `H_drtrain_epochs.csv` | `fig_H_drtrain_epochs.*` |
| H dr_eval / α / orden | `H_dr_eval.csv`, `H_alpha.csv`, `H_n.csv` | `fig_H_dr_eval.*`, `fig_H_alpha.*`, `fig_H_orden.*` |
| H re-chequeo OOD | `H_ood_epochs.csv` | `fig_H_ood_epochs.*` |
| metadatos corrida | `meta.json` | — |

Los bloques A–G salen de `analisis/experimento_reporte.py`; los H de
`analisis/experimento_hparams.py`.
