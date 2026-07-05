# Contexto completo del proyecto para redacción de tesis

> Documento AUTOCONTENIDO (snapshot julio 2026) para dar contexto a un
> asistente de escritura sin acceso al código. Contiene: problema, matemática,
> arquitecturas, protocolo experimental, TODOS los resultados numéricos,
> hallazgos, matices y glosario. Las figuras mencionadas (`fig_*.pdf/png`)
> existen en `Red/analisis/resultados/reporte/` y se pueden citar en LaTeX.

---

## 1. El problema

**Estabilización robusta de sistemas politópicos con certificado LMI.**
Un sistema lineal incierto se modela como politopo de N vértices:
ẋ = A(θ)x + B(θ)u, con (A(θ), B(θ)) ∈ conv{(A₁,B₁), …, (A_N,B_N)}.
Se busca una realimentación estática u = Kx que estabilice TODO el politopo
con tasa de decaimiento α. Condición suficiente (estabilidad cuadrática):
existen Q ≻ 0 (simétrica) e Y tales que, para cada vértice i,

  A_iQ + QA_iᵀ + B_iY + YᵀB_iᵀ + 2αQ ⪯ 0,  Q ⪰ εI,

y entonces K = YQ⁻¹. Esto es un SDP: resoluble por solvers exactos (CVXPY/SCS)
pero con costo que crece con N y n_x, y sin reutilización entre instancias.

**LMI-Net (enfoque amortizado, del paper base).** Una red neuronal
(encoder) mapea el sistema (A,B) a un vector ŷ que parametriza (Q,Y); una
capa de optimización diferenciable proyecta ŷ sobre el conjunto factible de
la LMI mediante iteraciones desenrolladas de Douglas-Rachford (DR). Se
entrena sin etiquetas (auto-supervisado): la pérdida se evalúa sobre el
certificado proyectado. En inferencia: un forward + un número fijo de
iteraciones DR (presupuesto `dr_eval`).

**Punto CRÍTICO de la arquitectura (clave para interpretar resultados):**
ŷ no es solo un "warm start" — es el punto que SE PROYECTA. El DR converge a
la proyección de ŷ sobre el conjunto factible, así que ŷ decide *a qué punto*
del conjunto factible se llega y qué tan rápido.

## 2. Contribuciones de esta tesis (sobre el paper base)

1. **Arquitectura invariante al nº de vértices N** (`vertices`): encoder
   tipo Deep Sets — un solo modelo sirve para politopos de cualquier N
   (el paper original fija N y n_u: un MLP por celda).
2. **Arquitectura invariante a N y equivariante al nº de actuadores n_u**
   (`actuadores`): un solo modelo cubre N∈{2..5} y n_u∈{1,2}.
   Ninguna arquitectura es invariante al ORDEN n_x (un modelo por orden).
3. **Diferenciación implícita** como alternativa al unrolling: backward por
   el teorema de la función implícita en el punto fijo del DR (memoria O(1)
   vs O(iters)); comparación controlada entre ambas.
4. **Registro de pérdidas extensible** + una pérdida nueva (`control_margen`)
   que corrige el modo de fallo fuera de distribución (ver hallazgo 4).
5. **Suite de benchmarking reproducible**: comparación sistemática vs CVXPY
   (factibilidad = verdad de terreno, tiempo, y calidad según cada criterio),
   barridos de pole-shift para casos borde, y diagnóstico de a qué se deben
   los fallos (encoder vs solver).

## 3. Arquitecturas comparadas

| nombre | invarianza | encoder | params (n=3) | backprop default |
|---|---|---|---|---|
| `vanilla` (fiel al paper) | ninguna (una red por N, n_u) | MLP 64×64 ReLU | 6 345 | implícito, 500 iters DR fijas, ε=1e-3, pérdida del paper |
| `vertices` | N | Deep Sets (suma sobre vértices) | 36 617 | unrolling-30 |
| `actuadores` | N + equivar. n_u | Deep Sets doble (vértices y actuadores) | 69 769 | unrolling-30 |

**Pérdidas** (todas auto-supervisadas, sobre el certificado proyectado (Q,Y)):
- `paper` (Ec. 21 del paper): minimizar logdet(Q) − β·λ_min(F(y)), β=100.
  λ_min(F) premia margen interior / castiga violación residual.
- `control`: minimizar trace(Q) + 0.1·‖Y‖²_F (volumen + esfuerzo).
- `control_margen` (NUEVA): control + η·relu(piso − λ_min(Q)), η=1, piso=0.05.
  Motivación: las otras dos premian Q pequeña ⇒ empujan el certificado a la
  frontera casi-singular del conjunto factible, donde K = YQ⁻¹ explota.
  El término castiga λ_min(Q) < piso sin tocar certificados con margen.

## 4. Datos y protocolo experimental

- **Base**: `DB_ssf_RS_500_c.mat` — 500 sistemas politópicos aleatorios por
  celda (n_x, n_u, N), con n_x∈{2,3,4,5}, n_u∈{1,2} (n_u=1 cubre n_x=2..5;
  n_u=2 cubre 3..5), N∈{2,3,4,5}. Cada sistema se normaliza por
  γ = max|entradas de A,B| (todo queda en [−1,1]).
- **Protocolo**: 150 sistemas/celda, partición 80/20 (seed=42), Adam,
  batch=16, lr=1e-3, α=0.01, ε=1e-5, 100 épocas (15 en los bloques lentos).
  Evaluación a `dr_eval`=1000 iteraciones DR salvo indicación.
- **Métrica principal**: % estabilizado = fracción del test donde
  max_i Re λ(A_i + B_iK) < 0 con K = YQ⁻¹ del certificado de la red.
- **Verdad de terreno**: CVXPY resuelve la MISMA LMI (solo factibilidad —
  devuelve el certificado factible más cercano a y=0, NO optimiza ningún
  criterio de desempeño). `cvxpy_pct` = techo: % de sistemas donde existe
  certificado con margen α.
- **Reproducibilidad**: la config base con la MISMA semilla dio
  85.8 / 87.5 / 90.0 % en tres corridas (no-determinismo BLAS multihilo).
  ⇒ diferencias < ~5 puntos son ruido; todos los efectos reportados son
  mayores.

Scripts: `analisis/experimento_reporte.py` (bloques A–G, ~10.5 min) y
`analisis/experimento_hparams.py` (bloques H, ~50 min). Un comando cada uno.

---

## 5. RESULTADOS EN DISTRIBUCIÓN

### 5.1 Escalera de arquitecturas (celda común n_x=3, N=2) — fig_C_arquitecturas

| arquitectura | % estab. | % decay α | train (s) | infer (ms/sys) |
|---|---|---|---|---|
| vanilla (receta paper) | **90.0** | 90.0 | 31.2 | 4.8 |
| vertices | 76.7 | 73.3 | 6.3 | 4.9 |
| actuadores | 80.0 | 73.3 | 6.6 | 4.9 |
| techo CVXPY | 93.3 | — | — | 7.3 |

La especialista gana su celda; las invariantes pagan ~10 puntos por
generalizar, pero UN modelo `actuadores` cubre las 8 celdas (N×n_u) con 90%
medio (ver 5.2), donde la vanilla necesitaría 8 modelos.

### 5.2 Pérdidas (actuadores, n=3, N=2..5, media) — fig_A_losses

| pérdida | % estab. | % red mejor que CVXPY en su propia loss | 
|---|---|---|
| control | **90.0** | 3.7 |
| control_margen | 87.5 | 3.7 |
| paper | 84.2 | 40.7 |
| techo CVXPY | 90.0 | — |

- `control` ALCANZA el techo de factibilidad en distribución.
- El 40.7% de `paper` no significa mejor red: el certificado de CVXPY
  (cercano a y=0) es casi óptimo para trace(Q) pero muy subóptimo para
  logdet−βλ_min. La comparación red-vs-CVXPY depende del criterio.
- `control_margen` cuesta ~2.5 puntos en distribución; su pago está en OOD.

### 5.3 Unrolling vs diferenciación implícita (15 épocas) — fig_B_backprop

| backprop | % estab. | train (s) | loss final |
|---|---|---|---|
| unrolling-30 | **80.0** | **5.6** | 0.0037 |
| implícito | 65.8 | 207.5 | **0.00009** |

El implícito optimiza la pérdida 40× mejor y estabiliza 14 puntos MENOS
(37× más lento). El gradiente exacto minimiza trace(Q) "demasiado bien":
lleva Q a la frontera casi-singular. El truncamiento del unrolling es
regularización implícita.

### 5.4 Iteraciones desenrolladas en train × épocas — fig_D_drtrain, fig_H_drtrain_epochs

| dr_train | 5 | 30 | 120 | 500 |
|---|---|---|---|---|
| 15 épocas | 79.2 | 80.0 | 76.7 | 66.7 |
| 100 épocas | **88.3** | 85.8 | **55.8** | 63.3 |

Dosis-respuesta: más proyección en train ⇒ mejor loss ⇒ PEOR estabilización,
y más épocas AMPLIFICAN el daño (120: 76.7→55.8; su loss llega a ~0 con Q
colapsada). La mejor celda de toda la matriz: **dr_train=5 + 100 épocas**
(88.3%, 12 s de train — también la más barata).

### 5.5 Otros hiperparámetros (fig_H_lr, fig_H_epochs, fig_H_batch, fig_H_dr_eval, fig_H_alpha, fig_H_orden)

- **Épocas** (n=3): 80.0 (15) → 80.8 (50) → 85.8 (100) → **89.2 (200)** →
  89.2 (400). Satura en ~200. OJO: en n=2, 400 épocas DEGRADARON lo
  en-distribución (90→78.3) — el óptimo depende del orden.
- **dr_eval** (mismo modelo, sin reentrenar): 37.5% (100 iters) → 82.5%
  (500) → 85.8% (1000) → **92.5% (5000)**, monótono. Costo 35 ms/sys a 5000
  — aún 4–14× más rápido que CVXPY en N≥3. Es la perilla calidad/latencia.
- **α**: 79.2% (α=0.001), 87.5% (0.01), **89.2% (0.05)**, 72.5% (0.1).
  Techo CVXPY: 95 / 90 / 57.5 / 37.5%. Dos efectos: α muy chico deja el
  certificado sin margen (daña); α=0.05 estabiliza MÁS que su propio techo
  (89.2 vs 57.5) — margen exigente en train = objetivo pro-robustez, aunque
  el margen estricto sea infactible. α=0.1: el conjunto factible se vacía.
- **Orden n_x**: 81.7 (2), **87.5 (3)**, 79.2 (4), 75.8 (5); techo ≥90%
  en todos. La brecha en n=4–5 es de datos/capacidad (dim(y) crece
  cuadrático con n_x), no de factibilidad.
- **lr**: plano en [3e-4, 1e-2] (87–88%). **batch**: óptimo suave 16–32.

### 5.6 Velocidad red vs CVXPY (inferencia batcheada, de A_losses_porN)

| N | red (ms/sys) | CVXPY (ms/sys) | speedup |
|---|---|---|---|
| 2 | 4.8 | 7–14 | 1.5–2.9× |
| 3 | 6.1 | 136 | 22× |
| 4 | 8.4 | 22 | 2.6× |
| 5 | 9.7 | 481 | **50×** |

La ventaja es de THROUGHPUT (lotes) y crece con el tamaño del SDP. Para UN
sistema suelto pequeño, CVXPY es competitivo (~4-7 ms) — ambas cifras deben
reportarse juntas (honestidad experimental).

---

## 6. RESULTADOS FUERA DE DISTRIBUCIÓN (casos borde)

Sistemas construidos a mano, desplazados hacia la inestabilidad, nunca vistos
en entrenamiento. En TODOS los puntos CVXPY confirma factibilidad
(`lmi_factible=True`): los fallos son del método, no del problema.

### 6.1 Los dos bancos de prueba

**(a) Pole-shift nominal** (N=1, n_x=3): A → A + s·I con
A = [[0,1,0],[0,0,1],[2,−3,1]], B = [0,0,1]ᵀ, s ∈ [−1, 2].

**(b) Politopo "del profesor"** (N=2, n_x=2, propuesto por el guía):
desplazamiento POR VÉRTICE A_i(δ) = A_i⁰ + δ·D_i con
A₁⁰=[[0,1],[−2,−2]], D₁=diag(2,1); A₂⁰=[[0,1],[−2,−3]], D₂=diag(−2,1);
B=[0,1]ᵀ genérica (igual en ambos vértices); δ ∈ [0, 2].
Geometría: D₁ desestabiliza el vértice 1 (abscisa hasta +3.41 en δ=2) y D₂
ESTABILIZA el vértice 2 (hasta −2.4): el politopo se ESTIRA 4δ en la entrada
(1,1) — una sola Q común debe cubrir ambos extremos. En δ=1, A₁ tiene
autovalores exactos {0, +1} (det = 2(δ−1)²).

**Métrica OOD**: `iters_min` = menor presupuesto DR ∈ {100, 250, 500, 1000,
2000, 4000, 8000} que estabiliza (una pasada con checkpoints). CAVEAT: la
trayectoria de DR no es monótona — `iters_min` solo es conclusivo junto con
`estabilizado_en_max=True` (hay "cruces transitorios" por la región estable,
visibles en la pérdida `paper`).

### 6.2 Nominal (modelos n=3 de 5.2) — fig_E_nominal

Las tres pérdidas estabilizan hasta s≈0.5 (con costo creciente: 100→8000
iters) y COLAPSAN todas en s≥0.75 (abscisa +1.47), pese a factibilidad
garantizada. `control_margen` con defaults NO ayuda aquí. El N=1 nominal es
el OOD más agresivo (el modelo solo vio politopos N∈{2..5}).

### 6.3 Politopo del profesor (modelos n=2) — fig_F_profesor (RESULTADO CENTRAL)

| δ (abscisa A₁) | control | control_margen | paper |
|---|---|---|---|
| 0–0.43 (−1 … −0.36) | 100 iters | 100–250 | 100–250 |
| 0.57–1.0 (−0.14 … +1.0) | 2000–4000 | **100–1000** | 500–4000 |
| 1.14–1.71 (+1.4 … +2.8) | 4000–8000 | **2000–4000** | falla desde δ=1.29 |
| 1.86–2.0 (+3.1 … +3.4) | **falla** | **4000 ✓** | falla |

`control_margen` ELIMINA el estancamiento: estabiliza TODO el barrido hasta
δ=2 con ≤4000 iteraciones. El tercer panel de la figura muestra el porqué:
`control` aterriza SIEMPRE en el peor autovalor = −0.0100… (exactamente −α,
la frontera); `control_margen` mantiene margen interior (−0.46) y lo cede
con gracia al crecer δ.

### 6.4 Aislamiento del fallo: ¿solver o encoder? — fig_G_aislamiento (DIAGNÓSTICO CLAVE)

La MISMA proyección DR sobre los MISMOS politopos normalizados, pero desde
ŷ ALEATORIO (mediana de 5 seeds) en vez del ŷ de la red:

| δ | DR puro (ŷ aleatorio) | red `control` | red `control_margen` |
|---|---|---|---|
| 1.29 | 100 | 8000 | 2000 |
| 2.00 | **100** | falla | 4000 |

**El DR puro estabiliza δ=2 en ~100 iteraciones donde la red entrenada con
`control` falla a 8000.** El cuello de botella NO es el solver: es la
dirección del ŷ del encoder. Entrenado con una pérdida que premia Q pequeña,
el encoder OOD apunta hacia una esquina casi-singular de la frontera del
conjunto factible: convergencia lentísima, K = YQ⁻¹ explosiva a presupuesto
corto (autovalores de lazo cerrado de +300 con 100 iters), aterrizaje en −α
exacto y estancamiento total en los δ extremos. Un ŷ aleatorio se proyecta a
un punto genérico y sale rápido. Hipótesis descartadas por este experimento:
"DR falla con polos cerca del eje imaginario" y "el estiramiento del politopo
rompe el DR" (el condicionamiento del certificado CVXPY sí empeora 1→63 con
δ, pero al DR puro le da igual).

### 6.4b Anatomía del estancamiento (trayectoria iteración a iteración) — fig_trayectoria_dr

Peor autovalor de lazo cerrado muestreado cada 20 iteraciones de DR (hasta
8000), politopo del profesor. Tres paneles que muestran GRÁFICAMENTE los
modos de fallo:
1. **Estancamiento** (ŷ de `control`): la trayectoria baja por mesetas
   escalonadas (zig-zag); en δ=2 queda en meseta POSITIVA (+0.79) que nunca
   cruza el cero — el fallo no es falta de iteraciones sino un punto de
   llegada malo.
2. **Cruce transitorio** (ŷ de `paper`, δ=1.57): la trayectoria cruza a la
   región estable de pasada (~88 checkpoints bajo cero) y SE DEVUELVE,
   terminando en +0.51 — prueba visual de la no-monotonicidad (caveat de
   `iters_min`).
3. **El ŷ decide** (mismo sistema δ=2, tres ŷ): `control` meseta positiva;
   `control_margen` cruza en ~2500 iters; ŷ ALEATORIO cruza en ~60 y queda
   en −0.28. Es la versión "de cerca" del experimento de aislamiento (6.4).
Detalle: picos tempranos de hasta +5·10⁴ (recortados en la figura) — K=YQ⁻¹
explota cada vez que Q pasa por casi-singular durante la convergencia.

### 6.4c Mapa de polos en el plano complejo — fig_plano_complejo

Vista geométrica de la LMI (estabilizar = todos los autovalores de lazo
cerrado en el semiplano izquierdo, pasada la línea −α). Coordenadas
normalizadas (escalado uniforme por γ, preserva el signo de Re λ). Tres
paneles: (A) lazo ABIERTO — el vértice 1 migra al semiplano derecho con δ,
el vértice 2 a la izquierda; (B) lazo CERRADO de la red `control` — los polos
se AGOLPAN sobre el margen −α (restricción activa) y en δ altos uno escapa al
RHP; (C) δ=2 con tres ŷ — `control` deja el polo crítico en Re=+0.79 (RHP,
FALLA), `control_margen` en −0.010 (sobre el margen, estable), ŷ ALEATORIO en
−0.28 (interior). Es el hallazgo 3/4 dibujado en el plano: el encoder
entrenado se sienta en el filo del margen (frágil OOD), un ŷ genérico factible
queda adentro. Pesos congelados en modelo_control_n2.pt / modelo_control_margen_n2.pt
(reproducible; el fallo marginal de control en δ=2 varía entre corridas sin
congelar). Mecanismo geométrico detrás de todos los fallos OOD.

### 6.5 ¿Más épocas cambian el cuadro OOD? — fig_H_ood_epochs

| pérdida | épocas | % en distrib. (n=2) | δ máx OOD | iters en δ=2 |
|---|---|---|---|---|
| control | 100 | 90.0 | 1.71 | falla |
| control | 400 | 78.3 | **2.00** | 8000 |
| control_margen | 100 | 88.3 | **2.00** | 4000 |
| control_margen | 400 | 86.7 | **2.00** | 4000 |

Matiz importante: 400 épocas TAMBIÉN rescatan el OOD de `control`… pagando
12 puntos en distribución y 5× más entrenamiento. `control_margen` domina:
robusto en ambos regímenes, sin ajuste, con ~2 puntos de costo.

---

## 7. HALLAZGOS PRINCIPALES (síntesis para la discusión)

1. **En distribución la red alcanza el techo de factibilidad** (90% = techo
   CVXPY con pérdida `control`; 92.5% con dr_eval=5000) con inferencia
   batcheada 22–50× más rápida que el SDP en N≥3. La ventaja es de
   throughput; para un sistema suelto CVXPY es competitivo.
2. **Optimizar mejor la pérdida EMPEORA la estabilización** — tres evidencias
   independientes: (i) implícito vs unrolling (loss 40× mejor, −14 puntos),
   (ii) curva dr_train 5→500 monótona al revés, (iii) la interacción con
   épocas la amplifica (120 iters @ 100 ép: 55.8%). Mecanismo: las pérdidas
   tipo volumen empujan Q a la frontera casi-singular. El truncamiento del
   unrolling es regularización implícita. La pérdida es un proxy, no la métrica.
3. **El modo de fallo OOD es el encoder, no la capa de optimización**
   (experimento de aislamiento, 6.4). En esta arquitectura ŷ decide a qué
   punto del conjunto factible se llega; OOD apunta a la esquina mala.
4. **Regularizar el aterrizaje arregla el OOD barato**: `control_margen`
   estabiliza todo el barrido del profesor (abscisa +3.4) donde las otras
   fallan, pagando ~2 puntos en distribución. Alternativa cara: 4× épocas
   (rescata `control` pero −12 puntos en distribución y más iteraciones).
   Con defaults no rescata el nominal n=3 (ajuste de η/piso por orden =
   trabajo futuro).
5. **Perillas que importan y que no**: dr_eval es la perilla calidad/latencia
   post-entrenamiento (37.5→92.5% de 100 a 5000 iters); α tiene zona útil
   [0.01, 0.05] con el efecto "margen exigente = robustez" (α=0.05 estabiliza
   89.2% > su techo 57.5%); épocas saturan (~200 en n=3) y pueden dañar
   (n=2); lr y batch, planos.
6. **Especialista vs generalista**: la vanilla del paper gana su celda (90%
   vs 80%) pero necesitaría un modelo por celda; la invariante cubre 8 celdas
   con uno. El pico de la invariante está en n=3 y cae con el orden (brecha
   de datos/capacidad, techo sigue ≥90%).
7. **Caveats metodológicos que la tesis debe declarar**: varianza entre
   corridas ±4 puntos con la misma semilla (BLAS multihilo); `iters_min`
   requiere `estabilizado_en_max=True` (trayectoria DR no monótona, cruces
   transitorios); "estabilizado" (Re λ<0) ≠ "factible con margen α" — por
   eso la red puede superar el "techo" (notas en 5.5-α y 5.2-N=5).

## 8. Inventario de figuras (todas en PDF listas para LaTeX)

| figura | contenido |
|---|---|
| fig_A_losses | barras % estab. por pérdida y N + techo CVXPY; % red mejor que CVXPY |
| fig_B_backprop | unrolling vs implícito: % estab., tiempo, memoria |
| fig_C_arquitecturas | escalera vanilla/vertices/actuadores + inferencia log vs CVXPY |
| fig_D_drtrain | % estab. y costo vs dr_train (15 ép) |
| fig_E_nominal | OOD nominal: abscisa, iters necesarias, peor autovalor @8000 |
| fig_F_profesor | OOD profesor: abscisas por vértice, iters por pérdida, autovalor @8000 |
| fig_G_aislamiento | DR puro (ŷ aleatorio) vs ŷ de la red, iters vs δ (log) |
| fig_trayectoria_dr | anatomía del estancamiento: trayectoria del peor autovalor por iteración de DR — meseta positiva, cruce transitorio (zig-zag), y "ŷ decide" |
| fig_plano_complejo | mapa de polos en el plano complejo: migración de lazo abierto, lazo cerrado agolpándose en −α, y δ=2 con tres ŷ (control en RHP, margen en el borde, aleatorio adentro) |
| fig_H_lr / H_epochs / H_batch | barridos 1-eje |
| fig_H_drtrain_epochs | interacción dr_train × épocas (el refuerzo del hallazgo 2) |
| fig_H_dr_eval | calidad y costo vs presupuesto de inferencia |
| fig_H_alpha | % estab. y techo CVXPY vs α |
| fig_H_orden | % estab. y techo vs n_x |
| fig_H_ood_epochs | re-chequeo OOD 100 vs 400 épocas |

## 9. Glosario rápido (para leer CSVs y pies de figura)

- **estabilizado / stable_pct**: max_i Re λ(A_i+B_iK) < 0 en test.
- **decay_pct**: además cumple Re λ ≤ −α.
- **cvxpy_pct / lmi_factible**: existe certificado con margen α (verdad de
  terreno, no implica que el certificado de CVXPY sea "bueno" según ningún
  criterio).
- **iters_min**: menor presupuesto DR que estabiliza ese sistema (válido si
  estabilizado_en_max=True).
- **peor_eig_max / peor_eig_max_iters**: peor autovalor de lazo cerrado al
  mayor presupuesto probado (8000).
- **loss_red vs loss_cvxpy**: valor de la pérdida elegida logrado por la red
  vs por el certificado factible de CVXPY, en los mismos sistemas.
- **dr_train / dr_eval**: iteraciones DR desenrolladas en entrenamiento /
  usadas por un forward de inferencia. **BUDGETS** no es un parámetro del
  modelo: es la lista de presupuestos que sondea `iters_to_stabilize`.
- **pole-shift**: A → A + s·I (todos los polos +s) o A_i → A_i + δ·D_i
  (dirección por vértice, formato del profesor).
