# Contexto del código — LMI-Net politópico

> Documento **autocontenido** para dar contexto a un asistente sin acceso al repositorio.
> Describe la arquitectura del código, qué hace cada módulo, el protocolo experimental
> completo y las convenciones de datos. Estado: **agosto 2026**.
>
> Sustituye a `CONTEXTO_TESIS.md` (julio 2026), que documentaba resultados de los
> experimentos piloto y quedó obsoleto: aquellos scripts están hoy en `legacy/` y el
> barrido E1–E9 los subsume por completo.

### Documentos que acompañan a este

Este archivo describe **el código y el protocolo**. Los resultados numéricos viven aparte,
a propósito: duplicarlos aquí garantizaría que se desincronicen. Para dar contexto completo
a un asistente, adjuntar los tres:

| Archivo | Qué contiene |
|---|---|
| `CONTEXTO_CODIGO.md` *(este)* | módulos, protocolo, convenciones de datos, trampas conocidas |
| `Red/analisis/resultados/barrido/REPORTE_BARRIDO.md` | rankings y efectos marginales de las 7 etapas, generado desde los shards |
| `Red/analisis/resultados/barrido/PENDIENTES_CAPITULO.md` | extracciones puntuales para el capítulo, con los cálculos de verificación |

El reporte se regenera con `python -m barrido.reporte` y refleja el estado de los shards en
disco; si las cifras de los tres discrepan, **el reporte manda**.

---

## 0. El problema en una página

Un sistema lineal incierto se modela como un **polítopo de N vértices**:

```
ẋ = A(θ)x + B(θ)u,    (A(θ), B(θ)) ∈ conv{(A₁,B₁), …, (A_N,B_N)}
```

Se busca **una** realimentación estática `u = Kx` que estabilice *todo* el polítopo con
tasa de decaimiento α. La condición suficiente (estabilidad cuadrática) es que existan
`Q ≻ 0` simétrica e `Y` tales que, para cada vértice i:

```
A_iQ + QA_iᵀ + B_iY + YᵀB_iᵀ + 2αQ ⪯ 0        (N bloques)
Q ⪰ εI                                          (1 bloque)
```

y entonces `K = Y Q⁻¹`. Es un SDP: resoluble exactamente (CVXPY/SCS), pero con costo
creciente en N y n_x y **sin reutilización entre instancias**.

**El enfoque amortizado.** Un codificador mapea el sistema `(A,B)` a un vector `ŷ` que
parametriza `(Q,Y)`; una **capa de proyección diferenciable** lleva `ŷ` al conjunto
factible mediante iteraciones de Douglas–Rachford (DR). El entrenamiento es
**auto-supervisado**: no hay etiquetas, la pérdida se evalúa sobre el certificado
proyectado. En inferencia: un forward más un presupuesto fijo de iteraciones DR.

**Punto crítico para interpretar todo lo demás:** `ŷ` no es solo un *warm start*. Es el
punto que se proyecta. El DR converge a la proyección de `ŷ` sobre el conjunto factible,
así que `ŷ` decide **a qué punto** del conjunto factible se llega, no solo cuán rápido.

**Contribución de la memoria sobre el trabajo base:** el paper original fija `(n_x, n_u, N)`
y usa un MLP por celda topológica. Aquí se añaden dos codificadores invariantes que cubren
familias enteras de topologías con una sola red.

---

## 1. Mapa del repositorio

```
Base de Datos/          .mat originales de MATLAB + lectores
Red/
  red/                  arquitecturas y solver DR
    core.py             LMICore: LMI, DR, decodificación y-(Q,Y)      253 líneas
    vanilla.py          línea base aplanada (fiel al paper)            67
    vertices.py         Deep Sets sobre vértices (invariante a N)      61
    actuators.py        Deep Sets sobre (vértice, actuador)            79
    backprop_unrolling.py   gradiente por desenrollado                 25
    backprop_implicit.py    gradiente por teorema de la función implícita  134
  entrenamiento/
    training.py         carga, las 6 pérdidas, bucle de entrenamiento  293
  pipeline/
    data_loader.py      ingesta y normalización del .mat               93
    read_database.py    conversión .mat → dataset (utilidad manual)    141
  barrido/              EL ARNÉS EXPERIMENTAL (ver §5)
  analisis/             benchmarks, sondas y reporte de la base
  tests/                pruebas del solver, del gradiente y de sensibilidad
legacy/                 12 experimentos piloto que el barrido subsume
```

---

## 2. `red/` — arquitecturas y solver

### 2.1 `core.py` · `LMICore`

Clase base de las tres arquitecturas. Contiene **toda** la matemática; las subclases solo
aportan el codificador.

```python
LMICore(n=3, m=1, N=2, alpha=0.1, epsilon=1e-5, dr_iters=100,
        sigma=0.01, backprop="unrolling", sigma_adaptativo=False)
```

**Parametrización del certificado.** El vector de decisión es
`y = [vech(Q), vec(Y)]` con

| | |
|---|---|
| `dim_Q` | n(n+1)/2 (parte triangular superior de Q, simétrica) |
| `dim_Y` | m · n |
| `dim_y` | dim_Q + dim_Y |

Para n_x=3, m=1: `dim_y = 6 + 3 = 9`. La reconstrucción usa el buffer `triu_indices`
(registrado con `register_buffer`, por lo que viaja con `model.to(device)`).

**`lmi_blocks(Q, Y, A_poly, B_poly, alpha, epsilon)`** construye los `N+1` bloques:

```
F_i     = −(A_iQ + QA_iᵀ + B_iY + YᵀB_iᵀ + 2αQ)   i = 1..N
F_{N+1} =  Q − εI
```

Factibilidad ⟺ todos los bloques son PSD. Es un helper **compartido** por el solver y por
`paper_loss`, para no duplicar la construcción.

**El solver Douglas–Rachford.** El estado es `s = [y_k | x_k]` de dimensión
`dim_y + (N+1)·n_x²` (36 para n_x=3, N=2). Métodos clave:

| Método | Qué hace |
|---|---|
| `_dr_precompute(A,B)` | devuelve `L, c, M_inv`; `M_inv = (I + LᵀL)⁻¹`, **no depende de σ** |
| `_dr_step_batch(...)` | una iteración de DR, batcheada |
| `_dr_state_step(s,...)` | la misma iteración en forma de estado apilado, para los VJP |
| `_dr_iterate(...)` | bucle con parada temprana opcional por tolerancia |
| `_dr_final_proj(...)` | proyección final sobre el subespacio afín C₁ |

La iteración **siempre arranca en `ŷ`** (`y_k = y_hat.clone()`), con cualquier σ.

**El papel de σ (anclaje).** Dentro de cada paso:

```
y_avg = (2σ·ŷ + y_k) / (2σ + 1)
```

Es la solución de `argmin_y { σ‖y − ŷ‖² + ½‖y − y_k‖² }`, o sea que **σ pondera el término
"quédate cerca de ŷ"**. No controla la inicialización sino cuánto se *retiene* la propuesta
en cada iteración. Pesos efectivos sobre ŷ: 0.99 % (σ=0.005), 1.96 % (σ=0.01),
16.7 % (σ=0.1), 66.7 % (σ=1.0).

**σ adaptativo** (`sigma_adaptativo=True`): `_dr_precompute` fija
`σ_i = σ_base / (1 + max_i‖A_i − Ā‖₂)` como tensor `(B,1)`. Funciona sin tocar la
matemática porque `M_inv` no depende de σ y las tres fórmulas que lo usan difunden.

**Configuración de la rama implícita** (atributos del modelo):

```
implicit_max_iters     = 4000     implicit_tol           = 1e-9
implicit_adjoint_iters = 1000     implicit_adjoint_tol   = 1e-10
implicit_ridge         = 1e-6
```

### 2.2 Las tres arquitecturas

| Módulo | Invariancias | ε por defecto | Entrada |
|---|---|---|---|
| `vanilla.py` | ninguna: una red por `(n_x, n_u, N)` | **10⁻³** (fiel al paper) | vector aplanado de dimensión fija |
| `vertices.py` | permutación e inserción de vértices (N libre), `n_u` fijo = 1 | 10⁻⁵ | conjunto de fichas `[vec(A_i), vec(B_i)]` |
| `actuators.py` | vértices **y** actuadores (`N` y `n_u` libres) | 10⁻⁵ | fichas `[vec(A_i), b_ij]`, dimensión `n²+n` |

> **Trampa conocida:** la diferencia de ε entre `vanilla` y las invariantes es un
> *confound* en los contrastes de arquitectura. Ver §8.

Ambos codificadores invariantes son Deep Sets: `φ` por ficha, agregación (`mean` por
defecto), `ρ` sobre el agregado. `vanilla` usa MLP 64×64 ReLU, backprop implícito y 500
iteraciones DR fijas, replicando el paper.

### 2.3 Las dos estrategias de gradiente

**`backprop_unrolling.py`** (25 líneas): corre `K_tr` iteraciones construyendo grafo. El
gradiente es la suma parcial de Neumann de `(I − J_s)⁻¹`.

**`backprop_implicit.py`** (134 líneas): forward hasta el punto fijo **sin grafo**
(memoria O(1) en iteraciones), backward por el teorema de la función implícita.

- El adjunto `(I − J_s)ᵀ w = rhs` se resuelve **matrix-free**, solo con productos
  vector-jacobiana (`torch.func.vjp` sobre *una* iteración de DR).
- La resolución es por iteración de punto fijo estilo DEQ:
  `w ← (J_sᵀ w + rhs) / (1 + ridge)`.
- `_adjoint_solve` devuelve `(w, iters_usadas, residual_final)`.

Ambas ramas **comparten la salida**: las dos terminan con `_dr_final_proj` antes de
decodificar `(Q,Y)`. Eso aísla el gradiente como única variable entre ellas.

---

## 3. `entrenamiento/training.py`

**Ruta de la base de datos.** `MAT_FILE` se resuelve en tiempo de import buscando, en
orden, `Red/DB_ssf_RS_500_c.mat` y `Base de Datos/DB_ssf_RS_500_c.mat`. Es la **única
fuente de verdad**: `barrido/data.py`, `analisis/benchmark_dr_vs_cvxpy.py` y
`db_report/extract_db.py` la importan de aquí.

**`load_vertices(order, inputs, vertices, mat, limit)`** devuelve una lista de tuplas
`(A, B)` en float64, con `A: (N, n_x, n_x)` y `B: (N, n_x, n_u)`. La normalización
(dividir por `γ = max|A,B|`) la hace `RobustControlMatlabDataset`.

**`split_items(items, frac=0.8, seed=42)`** — partición reproducible. La semilla es **fija
en 42**, independiente de la semilla de la corrida, para que el conjunto de prueba sea
comparable entre corridas.

### Las seis funciones de pérdida

Todas reciben `(Q, Y, A_poly, B_poly)` y devuelven un escalar a **minimizar**.

| Nombre | Fórmula | Tratamiento de la frontera |
|---|---|---|
| `paper` | `logdet(Q) − 100·λ_min(F)` | mixto: volumen contra margen, sin cota |
| `control` | `tr(Q) + 0.1·‖Y‖²_F` | **atrae** a la frontera, sin contrapeso |
| `control_margen` | `control + 1.0·relu(0.05 − λ_min(Q))` | añade piso espectral |
| `esfuerzo` | `‖K‖²_F` con `K = YQ⁻¹` | indiferente a la geometría |
| `condicionamiento` | `κ(Q) = λ_max/λ_min` | penaliza esquinas degeneradas |
| `margen_norm` | `−λ_min(F) + 0.1·tr(Q)` | **repele** la frontera |

Notas de implementación importantes:

- `paper_loss` usa `+logdet(Q)` con signo **positivo**: minimizarlo minimiza el volumen del
  elipsoide. El signo negativo fue un bug antiguo, no reintroducir.
- `margen_norm` **necesita** el término `μ·tr(Q)`: `F` es homogénea de grado 1 en `y`, así
  que sin fijar la escala se puede inflar `λ_min` gratis y la pérdida degenera.
- `esfuerzo` calcula `Kᵀ = Q⁻¹Yᵀ` con `torch.linalg.solve`, más estable que invertir Q.

El registro es extensible: `register_loss(name, factory, direction)` y
`get_loss_fn(name, model)`. `LOSSES_6` fija el orden canónico.

---

## 4. `pipeline/` — ingesta

`RobustControlMatlabDataset(mat_filepath, order, inputs, vertices)` lee el cell array
`BASE` de MATLAB, filtra por la celda topológica pedida y normaliza. La base tiene forma
`(n_x, n_u, N, casos) = (5, 2, 5, 500)` con **28 celdas no vacías** y **14 000 sistemas**.

`read_database.py` es una utilidad manual que convierte `.mat` a `.mat`/`.txt` derivados;
sus salidas están gitignoradas porque se regeneran.

---

## 5. `barrido/` — el arnés experimental

Es el corazón del capítulo de resultados. Principio de diseño: **todo se guarda en crudo**,
una fila por punto de evaluación; los promedios se calculan después, nunca al escribir.

### 5.1 `config.py` — ejes y etapas

```python
LOSSES   = 6 pérdidas          ALPHAS   = [0.001, 0.01, 0.05, 0.1]
ARCHS    = [vanilla, vertices, actuadores]    DR_TRAIN = [5, 30, 120, 500]
NX       = [2, 3, 4, 5]        SIZES    = [50, 150, 400]     SEEDS = [42, 123]
SIGMAS   = [0.005, 0.01, 0.1, 1.0]            EPSILONS = [1e-5, 1e-4, 1e-3, 1e-2]

EPOCHS = 400
EPOCH_CKPTS = (50, 100, 200, 400)
DR_EVAL     = (100, 250, 500, 1000, 2000, 4000, 8000)
```

**Configuración base:** `actuadores`, n_x=3, 150 sistemas, K_tr=30, α=0.01.

**`RunConfig`** es un dataclass congelado. `run_id()` es determinista y sirve de nombre de
shard y de carpeta de checkpoints:

```
E1__actuadores__nx3__paper__unrolling__dr30__a0p01__sz150__s42
```

`sigma`, `epsilon` y `sigma_adaptativo` son `Optional` con defecto `None` (= usar el
defecto de la arquitectura) y **solo se anexan al `run_id` cuando difieren del defecto**.
Eso preserva bit a bit los identificadores de las 443 corridas previas y mantiene válido el
`resume`.

**Distinción esencial entre tipos de eje:**

- **Ejes barridos** — multiplican el número de entrenamientos.
- **Puntos de control** — época y presupuesto de inferencia. Se obtienen *dentro* de una
  corrida y **no** multiplican el costo.

### 5.2 `data.py` — celdas y escenarios

Una **celda** es una topología `(n_u, N)`. Reglas por arquitectura:

| Arquitectura | Entrena en | S1 | S2 |
|---|---|---|---|
| `actuadores` | (1,2),(1,3),(1,4) | 20 % de esas 3 celdas | (1,5) ∪ (2,N) para todo N |
| `vertices` | (1,2),(1,3),(1,4) | 20 % de esas 3 celdas | **solo (1,5)** |
| `vanilla` | (1,2) | 20 % de esa celda | **vacío** |

Con n_x=3 y 150 sistemas por celda: `actuadores` entrena en 450, S1 = 300, S2 = **2500**;
`vertices` S2 = **500**; `vanilla` entrena en 150, S1 = 100.

> **No existe conjunto de validación.** La partición es 80/20 y el 20 % es S1.

`load_cell` está memoizado por proceso (`lru_cache`) y **falla ruidosamente** si la base no
existe, en vez de degradarse a "celda vacía".

### 5.3 `evaluation.py` y `metrics.py`

**Sondeo del presupuesto en una sola pasada.** `dr_eval_checkpoints` corre DR hasta
`max(milestones)` y en cada hito proyecta y mide. El ladder completo cuesta lo mismo que la
corrida más larga, no la suma.

**`_iters_min`** — menor hito de la grilla en que el sistema queda estable, `NaN` si nunca.
Se reporta junto a `estab_en_max` porque **la trayectoria de DR no es monótona**.

**`certificate_metrics`** devuelve, por sistema:

```
worst      max_i Re λ(A_i + B_iK)        stable     worst < 0
decay      worst ≤ −α                    margin     −worst
lam_min_F  λ_min(F(y*))   ← residual de la LMI, ≥0 == factible
lam_min_Q, kappa_Q         normK2, normKf        proj_dist  ‖ŷ − y*‖
```

### 5.4 `run.py` y `driver.py`

`run_one(cfg, out_root)` entrena **una sola vez** hasta `cfg.epochs` y, en cada checkpoint
de época, guarda el modelo y evalúa S1 y S2 con el ladder completo. Devuelve
`{status, n_rows, t_train_s, mem_peak_mb, ...}`.

> `t_train_s` cubre **solo el entrenamiento**. La evaluación se cronometra aparte
> (columna `t_batch_s` de los resultados) y pesa ~13 % adicional.

`driver.py` reparte las corridas en un pool de procesos, con **un hilo de PyTorch por
proceso** en CPU. Es **retomable**: salta toda corrida cuyo shard `meta/<run_id>` ya exista.

### 5.5 `ood_banks.py` y `ood_eval.py` — el escenario S3

Dos familias **sintéticas** (no salen de la base), de **un solo sistema por punto de rampa**:

- **Desplazamiento de polos**: `A(s) = A⁰ + sI`, N=1, n_x=3, 31 valores de `s ∈ [−1,2]`.
- **Polítopo de vértice marginal**: `A_i(δ) = A_i⁰ + δD_i`, N=2, n_x=2, 21 valores de
  `δ ∈ [0,2]`. En δ=1 un vértice tiene un polo sobre el eje imaginario.

`ood_eval.py` **no entrena**: carga checkpoints y los evalúa. `professor_mechanism`
instrumenta el mecanismo de fallo comparando la propuesta de la red contra `ŷ` aleatorio.

> Los porcentajes de S3 son fracciones de **modelos** que estabilizan la instancia, no de
> sistemas.

### 5.6 `reporte.py`, `estado.py`, `figuras.py`

`reporte.py` agrega los shards crudos a `REPORTE_BARRIDO.md`. Métrica de referencia: **%
estabilizado en S2, época 400, `dr_eval`=8000, promedio sobre semillas**. Los shards
anteriores a E7 no traen columnas `sigma`/`epsilon`: se rellenan con el defecto que esas
corridas usaron de hecho (ε depende de la arquitectura).

`estado.py` monitorea en vivo leyendo shards del disco. `figuras.py` produce las figuras del
capítulo.

---

## 6. `analisis/` — benchmarks y sondas

| Archivo | Qué hace |
|---|---|
| `benchmark.py` | API reutilizable; aporta `polytope_from_vertices`, `shift_poles`, `normalize_system`, `_cvxpy_solve` |
| `validate_projection.py` | banco de validación numérica; `cvxpy_projection` es la referencia exacta |
| `benchmark_dr_vs_cvxpy.py` | DR por lotes contra CVXPY/SCS, barriendo N y presupuestos |
| `fig_geometria.py` | corte 2D **computado** del conjunto factible con la trayectoria real de DR |
| `probe_jacobiano.py` | materializa `J_s` y mide el espectro y el solve adjunto |
| `db_report/` | caracterización de la base (descriptores, figuras, `analisis_db.tex`) |

---

## 7. Las etapas experimentales

**Factorial** = se cruzan todas las combinaciones (grilla llena). **Axial** = se fija una
configuración base y se mueve un eje a la vez (una cruz por el centro). Los promedios
marginales **solo son válidos dentro de una etapa factorial**.

| Etapa | Tipo | Corridas | Qué varía | Resultado principal |
|---|---|---|---|---|
| E0 | — | 4 | prueba de humo, 2 épocas | — |
| **E1** | **factorial** | 240 | pérdida × α × gradiente | K_tr=30 → 93.3 %; implícita → 59.0 %; α=0.01 óptimo |
| E2 | axial | 48 | orden n_x | sin degradación: 91.4 / 95.2 / 94.1 / 91.1 |
| E3 | axial | 36 | tamaño de datos | curva plana: 93.4 → 95.2 → 95.6 |
| E4 | axial | 119 | gradiente + arquitectura × orden | desenrollado 93.3 vs implícita 59.0 |
| E5 | solo eval. | — | deriva paramétrica | `margen_norm` colapsa a 39.3 % |
| **E7** | **factorial** | 192 | σ × ε | **σ es el eje de mayor efecto: 23.8 puntos** |
| E8 | axial | 24 | σ adaptativo | +2.7 (n_x=4), +4.8 (n_x=5) |
| E9 | axial | 48 | arquitecturas a ε igualado | ventaja de `vanilla`: 3.50 → 0.42–1.33 |

Total en la memoria: **443**. Con E7–E9: **707**.

**Mejor configuración conocida:** `margen_norm` + `actuadores` + desenrollado K_tr=30 +
α=0.01 + 400 sistemas → 97.4 % en S2, κ(Q)=1.67, `iters_min`=100.
Con σ=0.005 y ε=10⁻² la mejor celda de E7 llega a **98.11 %**.

---

## 8. Trampas conocidas

Cosas que un asistente debe saber antes de sacar conclusiones de estos datos:

1. **El piso de ruido es 1.3 puntos.** La dispersión entre las dos semillas tiene mediana
   0.60 y percentil 75 de 1.28 (174 configuraciones en S2). Diferencias menores son empates.

2. **Los marginales crudos mienten entre etapas.** La celda n_x=3 y `actuadores` absorben
   todo E1, incluidas sus configuraciones deliberadamente malas. El marginal dice que n_x=3
   es el peor orden (83.5 %) cuando es el mejor (95.2 %). Usar siempre subconjuntos
   controlados.

3. **S2 no es el mismo conjunto para todas las arquitecturas.** `actuadores` evalúa 2500
   sistemas, `vertices` solo 500 —y la celda más difícil—, `vanilla` ninguno. Comparar
   sobre la celda común `(1,5)`: ahí ambos codificadores son equivalentes (±0.5 puntos).

4. **ε difiere por arquitectura** (10⁻³ en `vanilla`, 10⁻⁵ en las invariantes), lo que
   confunde el contraste de arquitecturas si no se controla.

5. **S2 cumplió doble rol**: criterio de selección y escenario reportado. No hay fuga de
   datos, pero el 97.4 % es el máximo sobre el barrido, no una estimación insesgada.

6. **`iters_min` es condicional al éxito**: los sistemas que nunca estabilizan no entran en
   la mediana.

7. **La hipótesis del "gradiente implícito divergente" fue medida y descartada.** `J_s`
   tiene 18 de 36 autovalores exactamente en 1, pero el entrenamiento no recorre ese
   subespacio (peso 10⁻¹² frente a 10³). El mecanismo por el que truncar mejora la
   generalización **sigue abierto**.

---

## 9. Convenciones de datos

Los shards viven en `Red/analisis/resultados/barrido/<ETAPA>/`:

| Subcarpeta | Contenido | ¿Versionado? |
|---|---|---|
| `results/` | una fila por sistema × escenario × época × `dr_eval` | **no** (2.1 GB) |
| `models/` | checkpoints `epoch_*.pt` | **no** (832 MB) |
| `meta/` | una fila por corrida: tiempo, memoria, nº de sistemas | sí |
| `loss/` | curva de pérdida por época | sí |

Todo en parquet, con las columnas de configuración replicadas en cada fila para poder
filtrar y pivotar sin joins. `results/` y `models/` se regeneran corriendo el barrido.

---

## 10. Cómo correr

Desde `Red/`, con el entorno de `requirements.txt`:

```bash
python -m barrido.driver --stage ALL --workers 12     # E1–E4, retomable
python -m barrido.driver --stage E7,E8,E9 --workers 12
python -m barrido.estado --watch 60                   # progreso en vivo
python -m barrido.ood_eval --models analisis/resultados/barrido/E1/models
python -m barrido.reporte
python -m analisis.benchmark_dr_vs_cvxpy
python tests/test_solver_equivalence.py
python tests/test_sensibilidad_smoke.py
```

El barrido corre **en CPU**: las matrices son de a lo sumo 5×5 y la GPU no amortiza la
latencia de lanzamiento de núcleos.

---

## 11. Cambios respecto del estado de julio 2026

Para quien conozca la versión anterior del código:

**Etapas**
- `stage_E6` **desapareció**: se fusionó dentro de `stage_E4`, que ahora genera las 120
  corridas del duelo de gradiente más la escalera de arquitecturas en los cuatro órdenes.
  Los shards y checkpoints se migraron reescribiendo `stage` y `run_id`.
- Se añadieron `stage_E7` (σ × ε, factorial), `stage_E8` (σ adaptativo) y `stage_E9`
  (arquitecturas a ε igualado).

**Modelo**
- `LMICore` acepta `sigma_adaptativo`; `_sigma_por_dispersion` modula el anclaje por la
  dispersión del polítopo. Las tres arquitecturas propagan el parámetro.
- `RunConfig` acepta `sigma`, `epsilon` y `sigma_adaptativo` como ejes barribles.

**Correcciones**
- La ruta de la base estaba cableada en cuatro sitios; ahora la resuelve
  `training.MAT_FILE` y el resto la importa.
- `data.py::load_cell` convertía *cualquier* excepción en "celda vacía", de modo que una
  base ausente se reportaba como "sin celdas de entrenamiento". Ahora falla con un mensaje
  que dice dónde debería estar el archivo.
- `ood_eval.load_checkpoint` abortaba el recorrido completo al encontrar un checkpoint
  anterior al registro del buffer `triu_indices`. Ahora tolera buffers deterministas
  ausentes y falla ruidosamente ante cualquier otra clave.
- `reporte.py` rellena `sigma`/`epsilon` en shards antiguos con el defecto real de cada
  arquitectura.

**Repositorio**
- Los 12 experimentos piloto pasaron a `legacy/` (con los imports rotos, documentado).
- `requirements.txt` era un `pip freeze` de macOS que pedía `torch==2.12.1`, versión
  inexistente. Ahora son 9 dependencias directas con pisos de versión.
- `results/` y `models/` salieron del control de versiones.
