# Deep Sets vs. MLP: explicación a fondo del encoder N-invariante

*Documento didáctico para la Memoria de Título. Todas las dimensiones concretas usan la
topología del proyecto: $n=3$ estados, $m=1$ actuador, $N$ vértices (variable).*

---

## 0. La pregunta de fondo: ¿qué *es* la entrada?

Un sistema politópico es un **conjunto de $N$ vértices** $\{(A_i, B_i)\}_{i=1}^N$, donde
cada vértice tiene $A_i \in \mathbb{R}^{3\times3}$ y $B_i \in \mathbb{R}^{3\times1}$. Es
decir, cada vértice aporta $n^2 + n m = 9 + 3 = 12$ números.

Hay dos formas de darle esto a una red neuronal:

- **(MLP)** Aplanar TODO en un solo vector gigante y pasarlo por una red densa.
- **(Deep Sets)** Tratarlo como lo que es —un conjunto— y procesar **cada vértice por
  separado** con la misma función, luego resumir.

La diferencia entre ambas es la clave de por qué una sirve para un solo $N$ y la otra para
cualquiera.

---

## 1. El backbone MLP actual

El vector de entrada se construye concatenando todos los vértices aplanados:

```
features = [ vec(A_1), vec(B_1), vec(A_2), vec(B_2), ..., vec(A_N), vec(B_N) ]
           └────────────────────── 12·N números ──────────────────────┘
```

Para $N=2$ eso es un vector de $24$; para $N=5$, de $60$. La red es:

```
features ∈ R^{12N} ──Linear(12N→128)──LayerNorm──Mish──Linear(128→128)──...──Linear(128→9) ── ŷ
```

**Dónde vive el problema:** la primera capa es una matriz de pesos
$W_1 \in \mathbb{R}^{128 \times 12N}$. Cada uno de los $12N$ números de entrada tiene **su
propia columna de pesos**. Esto implica dos cosas fatales para $N$ variable:

1. **El tamaño de $W_1$ depende de $N$.** Una red entrenada con $N=2$ tiene
   $W_1 \in \mathbb{R}^{128\times24}$ y físicamente no puede recibir un vector de $36$
   (N=3). Hay que construir y entrenar **una red distinta por cada $N$**.
2. **El orden importa.** La red aprende un peso específico para "la entrada $(1,1)$ de $A$
   del vértice nº 1", otro distinto para "la del vértice nº 2", etc. Pero los vértices de un
   politopo son un **conjunto**: reordenarlos no cambia el problema. El MLP desperdicia
   capacidad aprendiendo (mal) esa simetría que debería ser gratis.

```
        vértice1   vértice2   vértice3 ...
        ┌──────┐   ┌──────┐   ┌──────┐
input:  │ 12   │   │ 12   │   │ 12   │   ← se concatenan en UN vector de 12N
        └──┬───┘   └──┬───┘   └──┬───┘
           └──────────┼──────────┘
                      ▼
            Linear(12N → 128)   ← W1 tiene un tamaño fijo atado a N,
                      ▼            y una columna distinta por cada posición
                    ...
                      ▼
                 ŷ ∈ R^9
```

---

## 2. Deep Sets: procesar el vértice, no el conjunto aplanado

La idea central de Deep Sets (Zaheer et al., 2017) es cambiar la **unidad de
procesamiento**: en vez de operar sobre el conjunto entero aplanado, se opera sobre **un
vértice a la vez**, con la **misma función compartida**, y recién al final se agregan.

```
   x_1 = [vec(A_1),vec(B_1)] ∈ R^12 ──┐
   x_2 = [vec(A_2),vec(B_2)] ∈ R^12 ──┤   φ (MLP compartida,
   x_3 = [vec(A_3),vec(B_3)] ∈ R^12 ──┤   la MISMA para todos)
        ...                           │        R^12 → R^128
   x_N = [vec(A_N),vec(B_N)] ∈ R^12 ──┘
              │  │  │        │
              ▼  ▼  ▼        ▼
            h_1 h_2 h_3 ... h_N      ∈ R^128 cada uno
              └──┬──┴──┬─────┘
                 ▼  POOLING simétrico (media)        ← aquí, y SOLO aquí, aparece N
            z = (1/N) Σ_i h_i        ∈ R^128         (operación SIN parámetros)
                 ▼
                 ρ (MLP)  R^128 → R^9
                 ▼
               ŷ ∈ R^9
```

En tu código (`lmi_net_deepsets.py`):

- **φ** = `Linear(12→128)→LayerNorm→Mish→Linear(128→128)→LayerNorm→Mish`
- **pool** = `mean` sobre el eje de los vértices
- **ρ** = `Linear(128→128)→LayerNorm→Mish→Linear(128→9)`

---

## 3. Las transformaciones, paso a paso (con dimensiones reales)

Sea un lote (batch) de $B$ sistemas. El tensor de entrada ya no es plano:

| Paso | Operación | Forma del tensor |
|---|---|---|
| Entrada | `A_poly`, `B_poly` | $(B, N, 3, 3)$ y $(B, N, 3, 1)$ |
| 1. Aplanar cada vértice | `reshape` | $A\!:(B,N,9)$, $B\!:(B,N,3)$ |
| 2. Concatenar A y B | `cat(dim=-1)` | $x:(B, N, 12)$ |
| 3. φ por vértice | `phi(x)` | $h:(B, N, 128)$ |
| 4. **Pooling** sobre vértices | `h.mean(dim=1)` | $z:(B, 128)$ |
| 5. ρ | `rho(z)` | $\hat y:(B, 9)$ |
| 6. Desempaquetar | `_y_to_matrices` | $Q:(B,3,3)$, $Y:(B,1,3)$ |
| 7. Proyección LMI | `_project_dr` (idéntico) | $Q,Y$ factibles |

**El mecanismo exacto del "peso compartido" (paso 3).** Cuando se aplica `nn.Linear(12,128)`
a un tensor de forma $(B, N, 12)$, PyTorch opera **solo sobre la última dimensión** y
**reutiliza la misma matriz de pesos** para todas las combinaciones de $(B, N)$. Es decir,
el mismo $\phi$ (con sus $1664$ pesos de la primera capa) se aplica a los $N$ vértices, sin
importar cuántos sean. Esto es exactamente análogo a cómo una **convolución** comparte su
kernel a lo largo de todas las posiciones de una imagen, lo que le permite procesar imágenes
de cualquier tamaño.

---

## 4. Las tres claves de "una sola arquitectura para todo $N$"

1. **φ opera por-vértice.** Su matriz de pesos es $\mathbb{R}^{128\times12}$: su tamaño
   depende de la dimensión de UN vértice ($12$), **nunca de $N$**. Da igual que haya 2 o 100
   vértices.

2. **Pesos compartidos entre vértices.** No hay "pesos del vértice 1" vs "pesos del vértice
   2": hay UN $\phi$ aplicado a todos. Esto elimina la dependencia del orden y del número.

3. **El pooling colapsa $N$ y no tiene parámetros.** La media
   $z=\frac1N\sum_i h_i$ convierte un conjunto de tamaño variable en un vector fijo de $128$.
   $N$ aparece **únicamente** en esta suma sin parámetros entrenables, así que jamás toca el
   tamaño de ninguna capa.

Resultado: el grafo de pesos (φ, ρ y todo lo que sigue) tiene tamaños **fijos**; lo único
que cambia con $N$ es cuántas veces se evalúa φ y sobre cuántos términos se promedia. Por eso
**la misma red (36 617 parámetros) acepta $N=2,3,4,5,\dots$** sin modificación alguna.

---

## 5. Permutación: de equivarianza a invarianza

- El paso φ es **equivariante a permutación**: si reordenas los vértices a la entrada, los
  $h_i$ salen reordenados igual (porque φ se aplica idéntico a cada uno).
- El **pooling** (media/suma/max) es una **operación simétrica**: el resultado no depende del
  orden de sus argumentos. Aquí la equivarianza se convierte en **invarianza**:
  $z$ es el mismo sin importar cómo se ordenen los vértices.
- ρ recibe $z$ ya invariante, así que $\hat y$ es invariante a permutación.

Lo verificamos numéricamente: permutar los vértices cambia la salida en $\sim10^{-16}$ (error
de redondeo de la máquina, es decir, exactamente cero). Esto no es solo conveniente: es el
**sesgo inductivo correcto**, porque el conjunto factible de la LMI es idéntico bajo cualquier
reordenamiento de los vértices. El MLP tiene que *aprender* esta simetría con datos (y no la
garantiza); Deep Sets la trae **incorporada por construcción**.

---

## 6. ¿Por qué el MLP no sirve ni "rellenando" (padding)?

Una tentación es fijar $N_{\max}=5$ y rellenar con ceros los vértices faltantes. No funciona:

- **Sigue sin ser invariante a permutación:** el MLP asigna pesos distintos a cada ranura, así
  que un vértice en la posición 1 se trata distinto que en la 3.
- **Los ceros mienten:** un "vértice de relleno" $A=0, B=0$ es, para la red, un vértice real
  con esas matrices — una restricción LMI espuria, no una ausencia.
- **No extrapola:** con $N_{\max}=5$ jamás podrías evaluar un sistema con $N=6$. Deep Sets sí
  (de hecho lo demostramos: entrenado en $N\in\{2,3,4\}$ estabiliza el 85 % de los $N=5$).

---

## 7. La elección del pooling: media vs. suma vs. máximo

Las tres son simétricas (todas dan invarianza). La diferencia está en la escala:

- **Suma** $\sum_i h_i$: su magnitud **crece con $N$** (la suma de 5 vértices es ~2.5× la de
  2). Al entrenar con $N$ mezclados, ρ vería entradas de escalas muy dispares.
- **Media** $\frac1N\sum_i h_i$: normaliza por $N$, manteniendo la **escala constante** sin
  importar cuántos vértices haya. Por eso es la elección natural para entrenamiento con $N$
  mixto, y es la que usamos.
- **Máximo** (elemento a elemento): captura "el vértice más extremo en cada característica";
  robusto a *outliers*. Alternativa válida a contrastar.

---

## 8. Garantía teórica: el teorema de Deep Sets

No es un truco de ingeniería. El teorema de representación (Zaheer et al., 2017) dice:

> Toda función continua **invariante a permutación** $f(\{x_1,\dots,x_N\})$ puede escribirse
> como $\rho\!\left(\sum_i \phi(x_i)\right)$ para ciertas $\phi,\rho$; y recíprocamente, toda
> función de esa forma es invariante a permutación.

Es decir, la arquitectura Deep Sets es un **aproximador universal de exactamente la clase de
funciones que necesitamos** (las que no dependen del orden de los vértices). La cabeza ρ +
pooling + φ no nos limita: puede representar cualquier mapa $\{(A_i,B_i)\}\mapsto(Q,Y)$ que
respete la simetría del problema.

---

## 9. Conteo de parámetros (la comparación contundente)

| | Primera capa | ¿Depende de $N$? | Total (topología actual) |
|---|---|---|---|
| **MLP** | `Linear(12N → 128)` = $1536N+128$ | **Sí** (3200 si N=2; 7808 si N=5) | ~21 k, **distinto por N** |
| **Deep Sets** | `Linear(12 → 128)` = $1664$ | **No** | **36 617, único para todo N** |

El MLP necesita 4 modelos para cubrir $N\in\{2,3,4,5\}$; Deep Sets, uno solo.

---

## 10. Por qué la arquitectura "calza" con la matemática

Recordando la asimetría del problema: la **entrada** es un conjunto de tamaño variable y sin
orden; la **salida** es un certificado **único y de tamaño fijo** $(Q,Y)$ —porque se busca una
sola función de Lyapunov común a todo el politopo (estabilizabilidad cuadrática)—. El mapa a
aprender es literalmente *conjunto variable $\to$ vector fijo, invariante a permutación*. Deep
Sets **es** esa familia de funciones. La arquitectura no se adapta forzadamente al problema:
**refleja su estructura matemática**.

---

### Resumen en una frase

El MLP aplana los $N$ vértices en un vector cuyo tamaño y orden lo atan a un $N$ fijo; Deep
Sets aplica una misma red $\phi$ a **cada vértice por separado** y los resume con un
**promedio simétrico**, de modo que el tamaño de los pesos no depende de $N$ y el resultado no
depende del orden — una sola arquitectura, invariante a permutación y a cardinalidad, que
además generaliza a números de vértices nunca vistos.
