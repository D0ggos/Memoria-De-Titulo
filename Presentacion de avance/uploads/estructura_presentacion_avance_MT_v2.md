# Estructura — Presentación de Avance de Memoria de Título (v2)
**Pedro Pablo Muñoz Solano — Universidad de Concepción**
*Profesor Guía: Hugo Garcés Hernández — Profesor Co-Guía: Jonathan Palma Olate*

> **Cambios respecto a v1:**
> 1. Atribución correcta de LMI-Net al paper de Tang, Goertzen & Azizan (MIT, 2026); el aporte propio se posiciona como aplicación + análisis + extensión.
> 2. Eliminada la noción de "tiempo real estricto"; reemplazada por **descomposición offline–online**.
> 3. Conservada la idea de **meta-controlador** y conectada explícitamente con la propuesta original.
> 4. Agregado un slide dedicado al **problema de dimensionalidad fija** como línea de trabajo en curso.
> 5. **Marco teórico ampliado** (slides 4–7) para los evaluadores (uno matemático/DL, otro de algoritmos/optimización).
> 6. **Trabajo futuro corregido**: Dykstra reformulado como "una alternativa entre varias" y criterios de desempeño filtrados a los que son **viables con la BD UNICAMP** (sin H∞).

---

## Resumen del diseño

- **Duración estimada:** 18–22 minutos de exposición + 5–10 min de preguntas.
- **N° de slides recomendado:** 23 slides (+ 6 de respaldo).
- **Estilo visual sugerido:** sobrio, técnico-académico. Azul UdeC institucional; acentos en cian/verde para resultados positivos, ámbar/rojo para limitaciones/riesgos.
- **Audiencia objetivo:** un evaluador experto en **matemáticas y Deep Learning** (querrá rigor en ecuaciones y arquitectura) y otro en **algoritmos y optimización** (querrá rigor en el splitting y complejidad). Cada slide técnico debe satisfacer a ambos sin perder al otro.
- **Principio de oro:** la pantalla apoya, no reemplaza al expositor. Ecuaciones grandes y limpias, prosa mínima.

Cada slide tiene cuatro componentes:
1. **Título** — lo visible arriba.
2. **Contenido en pantalla** — bullets, ecuaciones, diagramas.
3. **Notas para el orador (qué decir)** — guion.
4. **Criterio(s) de la rúbrica cubiertos** — mapeo explícito.

---

## Bloque I — Apertura (1 slide)

### **Slide 1 — Portada**

**Contenido en pantalla:**
- Logo UdeC + Facultad de Ingeniería + DIICC.
- **Título propuesto de la tesis** (refleja la evolución):
  > *"Síntesis offline–online de controladores robustos para sistemas LPV politópicos: un meta-controlador basado en Deep Learning con garantías de estabilidad por construcción"*
- Subtítulo: **Presentación de Avance — Memoria de Título**.
- Pedro Pablo Muñoz Solano.
- Profesor Guía: Hugo Garcés Hernández. Co-Guía: Jonathan Palma Olate.
- Mayo 2026.

**Notas para el orador:**
> *"Buenas tardes. Mi nombre es Pedro Muñoz, estudiante de Ingeniería Civil Informática. Hoy presentaré el avance de mi memoria de título, que aborda un problema en la intersección entre la teoría de control robusto, el aprendizaje profundo y los métodos de proyección convexa: cómo diseñar un meta-controlador que aprenda a sintetizar ganancias estabilizadoras para sistemas con incertidumbre paramétrica, garantizando por construcción el cumplimiento de las restricciones matriciales que certifican estabilidad."*

**Criterios cubiertos:** Presentación de apoyo (5%), Comunicación (5%).

---

## Bloque II — Introducción y Planteamiento del Problema (3 slides)

### **Slide 2 — Contexto: Sistemas Ciberfísicos y la exigencia de estabilidad**

**Contenido en pantalla:**
- Definición breve: **CPS = computación + comunicación + control + procesos físicos.**
- Tres tarjetas (iconos + caption corto):
  1. **Apagón Norteamérica 2003** — 50 M sin suministro por inestabilidad de control en cascada.
  2. **Ciberataques a actuadores** (FDI / DoS) — modifican parámetros efectivos de la planta.
  3. **Microgrids, vehículos autónomos, plantas con desgaste** — parámetros que cambian en milisegundos.
- Idea de cierre: *"Una mala decisión de control no es un bug: es daño físico."*

**Notas para el orador:**
> *"Los sistemas ciberfísicos están en todas partes: redes eléctricas, vehículos autónomos, plantas industriales. Lo que los distingue es que un error en software se traduce en daño físico real. El apagón del 2003 en Norteamérica no fue un fallo eléctrico clásico: fue una inestabilidad de control mal coordinado. Hoy, además, los CPS enfrentan ciberataques que modifican los parámetros efectivos de la planta. Por eso, garantizar estabilidad no es un lujo: es un requisito."*

**Criterios cubiertos:** Introducción/contexto (5%), Comunicación.

---

### **Slide 3 — Problema concreto: sistemas LPV politópicos y la barrera computacional**

**Contenido en pantalla — dos columnas:**

*Izquierda (geometría):* polítopo 2D con vértices $A_1, A_2, ..., A_N$ y una trayectoria paramétrica $\xi(t)$ moviéndose en su interior.

*Derecha (matemática):*
- Sistema LPV: $\dot{x}(t) = A(\xi(t))\,x(t) + B(\xi(t))\,u(t)$
- Estabilidad cuadrática:
  $$P \succ 0, \quad A(\xi)^T P + P A(\xi) \prec 0, \;\forall\, \xi \in \Xi$$
- Reducción politópica: si la dependencia es afín en $\xi$, infinitas restricciones se reducen a $N$ LMIs evaluadas en los vértices.

*Cierre (resaltado):* **"Resolver $N$ LMIs en línea con métodos de punto interior escala como $\mathcal{O}(m^3)$; impracticable para CPS modernos."**

**Notas para el orador:**
> *"Un sistema LPV es un sistema lineal cuyos parámetros varían en el tiempo según un vector $\xi(t)$ que vive en un polítopo. Gracias a la convexidad de ese polítopo y a la linealidad de las LMIs, garantizar estabilidad para todo $\xi$ se reduce a satisfacer una LMI por cada vértice. Sin embargo, esos $N$ problemas matriciales acoplados se resuelven con métodos de punto interior cuya complejidad cúbica los hace inviables cuando el sistema cambia varias veces por segundo."*

**Criterios cubiertos:** Planteamiento (10%), Aprendizaje autónomo (10%).

---

### **Slide 4 — Pregunta de investigación**

**Contenido en pantalla — una sola pregunta centrada:**
> **"¿Es posible aprender, en una fase offline barata, un mapping de sistemas LPV politópicos a controladores robustos válidos, de modo que la evaluación online se reduzca a una inferencia rápida que cumpla *por construcción* todas las restricciones matriciales de estabilidad?"**

Tres palabras clave subrayadas: **aprende**, **por construcción**, **offline–online**.

**Notas para el orador:**
> *"En una sola pregunta: ¿puedo pagar offline el costo de aprender, para que online cada decisión sea casi gratuita y, lo más importante, matemáticamente válida? Esa descomposición offline–online es la idea filosófica detrás del trabajo. Y la frase 'por construcción' marca la diferencia con cualquier red neuronal convencional aplicada a control: las restricciones no se penalizan, se imponen geométricamente."*

**Criterios cubiertos:** Planteamiento (10%), Pensamiento crítico (5%).

---

## Bloque III — Marco Teórico Esencial (4 slides)

### **Slide 5 — Lyapunov: la energía virtual y su gradiente**

**Contenido en pantalla — diagrama + ecuaciones:**

- Gráfico 3D estilizado de una función de Lyapunov $V(x) = x^T P x$ como un "tazón" cuadrático.
- Una trayectoria que desciende por el tazón hasta el origen.

**Ecuaciones esenciales:**
- Candidata: $V(x) = x^T P x$, con $P \succ 0$.
- Disipación: $\dot V(x) = x^T (A^T P + P A) x < 0$, $\forall x \neq 0$.
- $\Rightarrow$ Condición algebraica equivalente: $A^T P + P A \prec 0$.

*Cierre:* **"Una matriz $P$ define una métrica; cumplir Lyapunov significa que esa métrica decrece a lo largo de toda trayectoria."**

**Notas para el orador:**
> *"El método directo de Lyapunov es la piedra angular. La idea, debida a Aleksandr Lyapunov hace más de un siglo, es brillante: en vez de resolver la ecuación diferencial, postulo una 'energía virtual' del sistema —$V(x) = x^T P x$— y exijo que esa energía disipe en todo momento. Si esa derivada temporal es negativa para todo estado, el sistema es asintóticamente estable. Y como la derivada se expresa algebraicamente, todo el problema dinámico se transforma en una desigualdad matricial."*

**Criterios cubiertos:** Aprendizaje autónomo (10%), Pensamiento crítico (5%).

---

### **Slide 6 — LMIs y el problema de síntesis: BMI → LMI por congruencia**

**Contenido en pantalla — flujo de tres pasos:**

**Paso 1 — Lazo cerrado:** Con realimentación $u = Kx$, la dinámica deviene $\dot x = (A + BK)x$.

**Paso 2 — Condición bilineal (BMI, NO convexa):**
$$(A + BK)^T P + P (A + BK) \prec 0$$
Las incógnitas $P$ y $K$ aparecen multiplicadas. No es una LMI.

**Paso 3 — Transformación de congruencia:** se definen $Q = P^{-1}$ y $Y = K Q$. Multiplicando por $Q$ a ambos lados:
$$A Q + Q A^T + B Y + Y^T B^T \prec 0, \quad Q \succ 0$$

Esta SÍ es lineal en $(Q, Y)$ → **LMI convexa**.

Para LPV politópico: la condición se replica en cada vértice $(A_i, B_i)$ con un único par $(Q, Y)$ común.

**Notas para el orador:**
> *"Si quiero diseñar el controlador, no basta con verificar estabilidad: tengo que encontrar simultáneamente la matriz $P$ y la ganancia $K$. Pero al multiplicar el lazo cerrado, aparecen productos $PK$ y $K^T P$. Eso es una BMI, bilineal, no convexa, intratable. La transformación de congruencia —cambiar variables a $Q = P^{-1}$ e $Y = KQ$— linealiza la condición y la convierte en una LMI convexa. Este truco, conocido al menos desde Boyd y colegas en 1994, es lo que hace tratable el problema."*

**Criterios cubiertos:** Aprendizaje autónomo (10%), Pensamiento crítico (5%).

---

### **Slide 7 — Proyección sobre la intersección de conjuntos convexos**

**Contenido en pantalla:**

- Definición de **proyección euclidiana** sobre un conjunto convexo $\mathcal{C}$:
  $$\Pi_{\mathcal{C}}(\hat y) = \arg\min_{z \in \mathcal{C}} \tfrac{1}{2}\|\hat y - z\|_2^2$$

- Diagrama 2D: un conjunto $\mathcal{C}$ curvado, un punto $\hat y$ fuera, y la flecha mínima hacia $\Pi_\mathcal{C}(\hat y)$.

- **Idea clave:** si $\mathcal{C} = C_1 \cap C_2$, proyectar sobre $\mathcal{C}$ directamente puede ser intratable, pero proyectar sobre cada $C_i$ por separado puede ser barato.

- **Splitting methods** (Douglas–Rachford, Dykstra, ADMM, ...) explotan esa descomposición: alternan proyecciones sobre cada subconjunto y combinan los resultados con un esquema de relajación o corrección.

**Notas para el orador:**
> *"La pieza algorítmica que conecta Deep Learning con LMIs es la proyección. Proyectar un punto sobre un conjunto convexo es, formalmente, encontrar el punto más cercano dentro del conjunto. Si el conjunto es complejo, la proyección directa no tiene forma cerrada. Pero si puedo descomponerlo en dos conjuntos más simples sobre los cuales sí sé proyectar, los métodos de splitting hacen la magia: alternan proyecciones individuales y convergen al objetivo común. Esta familia de algoritmos —Douglas–Rachford es uno entre varios— es lo que permite que la capa sea diferenciable y entrenable."*

**Criterios cubiertos:** Aprendizaje autónomo (10%), Estado del arte (10%), Curiosidad científica que vincula disciplinas (5%).

---

### **Slide 8 — Estado del arte: redes neuronales con restricciones duras**

**Contenido en pantalla — tabla comparativa:**

| Enfoque | Referencia | Tipo de restricción | Garantía | Limitación |
|---|---|---|---|---|
| Solvers IPM clásicos (CVXPY/SCS) | Boyd et al. 1994 | LMI / SDP general | ✅ Exacta | Lento (ms–s) |
| Warm-start aprendido | **Sambharya et al., 2023** | Cualquier QP | Vía solver | Sigue requiriendo solver online |
| RAYEN | **Tordesillas et al., 2023** | Convexas lineales/cuadráticas | ✅ Por construcción | No aborda LMIs matriciales |
| HardNet | **Min & Azizan, 2024** | Restricciones generales | ✅ Por construcción | No específica para LMIs acopladas |
| πNet (ortogonal projection) | **Grontas et al., ICLR 2026** | Convexas con proyección ortogonal | ✅ Por construcción | Foco general, no LMIs |
| **LMI-Net** | **Tang, Goertzen & Azizan, MIT, arXiv 2026** | **LMIs por descomposición afín + PSD** | ✅ Por construcción | Dimensionalidad fija en la red |

*Cierre (resaltado):* **"LMI-Net es la primera capa diferenciable que aborda directamente restricciones LMI matriciales input-dependent. Esta memoria adopta esa arquitectura y la aplica al problema politópico LPV."**

**Notas para el orador:**
> *"El campo ha evolucionado muy rápido desde 2023. Sambharya propuso aprender un warm-start para acelerar solvers; Tordesillas con RAYEN y Min con HardNet introdujeron capas de proyección diferenciable para restricciones convexas escalares; Grontas y colegas, en ICLR 2026, generalizan a proyecciones ortogonales. Pero ninguno aborda LMIs matriciales acopladas, que son el lenguaje nativo del control robusto. LMI-Net, del grupo de Azizan en MIT publicado en arXiv en abril de 2026, es la primera arquitectura diseñada específicamente para esta clase de restricciones. Es la herramienta que adopto en mi memoria; mi aporte no es haberla inventado, sino llevarla al dominio de LPV politópicos y validarla sobre la base de datos UNICAMP, con todo el análisis crítico que eso conlleva."*

**Criterios cubiertos:** Estado del arte con 4+ refs post-2021 (10%), Selección de info pertinente (5%), Curiosidad científica (5%).

---

## Bloque IV — Objetivos y Posicionamiento (3 slides)

### **Slide 9 — De la propuesta original al estado actual: evolución de la idea**

**Contenido en pantalla — esquema de dos columnas (antes / ahora):**

| **Propuesta original (inicio)** | **Estado actual (avance)** |
|---|---|
| Meta-controlador que mapea **estados** → ganancias en tiempo real | Meta-controlador que mapea **plantas** (matrices del sistema) → parametrización $(Q, Y)$ válida |
| Sistemas **discretos** ($x_{k+1} = Ax_k + Bu_k$) | Sistemas **continuos** ($\dot x = Ax + Bu$); LMI continua de Lyapunov |
| Penalización LMI tipo *soft constraint* | Restricción **dura** por construcción vía capa de proyección LMI-Net |
| Arquitectura abierta | Arquitectura concreta: **MLP + capa Douglas–Rachford** (Tang et al., 2026) |
| Esquema online único | **Descomposición offline–online** explícita |

*Cierre:* **"El espíritu del meta-controlador se conserva: una red que opera *dentro* del dominio factible. Lo que cambió fue cómo se garantiza ese dominio: de penalización suave a proyección dura."**

**Notas para el orador:**
> *"Quiero ser transparente con la evolución del trabajo. La propuesta inicial planteaba un meta-controlador con penalización suave de la LMI. Durante la investigación bibliográfica encontré que el grupo de Azizan en MIT había publicado, en abril de 2026, una capa diferenciable que impone LMIs como restricción dura. Eso transformó la tesis: en lugar de penalizar, ahora proyecto. La idea filosófica del meta-controlador —una red que opera dentro del dominio factible— se conserva, pero ahora con un mecanismo formal mucho más fuerte. También migré de sistemas discretos a continuos para alinear con la base de datos de UNICAMP, que es el benchmark estándar en el campo."*

**Criterios cubiertos:** Pensamiento crítico (5%), Innovación (5%), Aprendizaje autónomo (10%).

---

### **Slide 10 — Objetivo General**

**Contenido en pantalla — bloque centrado:**
> **Objetivo General**
> *Implementar y validar empíricamente un **meta-controlador basado en Deep Learning** que sintetice controladores estabilizadores para una familia de sistemas LPV politópicos en tiempo continuo, **adoptando la capa de proyección diferenciable LMI-Net** (Tang, Goertzen & Azizan, 2026) sobre la división de Douglas–Rachford para garantizar por construcción el cumplimiento de las desigualdades matriciales lineales de Lyapunov.*

**Notas para el orador:**
> *"El objetivo general es claro y honesto sobre la atribución: implementar y validar empíricamente, adoptando LMI-Net como capa de garantía. La aplicación al problema específico de LPV politópicos sobre la base de datos UNICAMP, el análisis de limitaciones y la propuesta de extensiones constituyen el aporte original de esta memoria."*

**Criterios cubiertos:** Objetivos (10%), Sección Objetivos en avance (5%).

---

### **Slide 11 — Objetivos Específicos**

**Contenido en pantalla (5 bullets):**

1. **Construir un pipeline reproducible** de extracción y preprocesamiento desde la base de datos UNICAMP (formato `.mat`) hacia tensores PyTorch en precisión doble, indexando sistemas por dimensión, número de actuadores y vértices del polítopo.

2. **Implementar la arquitectura LMI-Net** sobre PyTorch con la capa de proyección Douglas–Rachford diferenciable, configurada para sistemas LPV politópicos en tiempo continuo.

3. **Formular criterios de desempeño físicos compatibles con los datos disponibles**: maximización de la tasa de decaimiento $\alpha$ y minimización del volumen del elipsoide invariante $-\log\det Q$.

4. **Evaluar empíricamente** el meta-controlador en términos de: (i) tasa de estabilización sobre sistemas no vistos, (ii) cumplimiento del margen $\alpha$ exigido, (iii) tiempo de inferencia comparado con CVXPY/SCS.

5. **Caracterizar críticamente las limitaciones** observadas (gap de entrenamiento, condicionamiento numérico, rigidez dimensional) y delinear extensiones algorítmicas para futuras iteraciones.

**Notas para el orador:**
> *"Cinco objetivos específicos. Los cuatro primeros cubren el ciclo técnico completo: datos, arquitectura, formulación de pérdidas y evaluación. El quinto es deliberadamente analítico: una memoria de título debe ser honesta sobre lo que aún no funciona perfecto, y abrir el camino. Más adelante mostraré que ese quinto objetivo no es una cláusula de escape: ya tengo limitaciones cuantificadas y caminos de mitigación concretos."*

**Criterios cubiertos:** Objetivos (10%), Pensamiento crítico (5%).

---

## Bloque V — Diseño de la Solución (4 slides)

### **Slide 12 — La descomposición offline–online**

**Contenido en pantalla — flujo horizontal en dos bloques:**

```
┌──────────────────────────────┐     ┌──────────────────────────────┐
│        OFFLINE (caro)        │     │       ONLINE (barato)        │
│                              │     │                              │
│  Base de datos UNICAMP       │     │  Nueva planta (Aᵢ, Bᵢ)       │
│  (500 sistemas RS)           │     │           ↓                  │
│           ↓                  │     │  Inferencia MLP              │
│  Entrenamiento auto-supervisado │  │           ↓                  │
│  (50 épocas, 30 iter DR)     │     │  Capa de proyección DR       │
│           ↓                  │     │           ↓                  │
│  Modelo θ* aprendido         │ ──▶ │  (Q, Y) factible ✓           │
│                              │     │           ↓                  │
│                              │     │  K = Y · Q⁻¹                 │
│                              │     │                              │
│  Costo: horas en GPU         │     │  Costo: μs–ms por instancia  │
└──────────────────────────────┘     └──────────────────────────────┘
```

*Cierre:* **"El paradigma no es 'tiempo real estricto'; es amortización: pagar una vez offline para servir muchas veces online."**

**Notas para el orador:**
> *"Quiero ser preciso con la terminología: este enfoque no es 'tiempo real' en el sentido más estricto de la palabra. Es lo que el paper original llama 'descomposición offline–online'. Se invierte una cantidad significativa de cómputo offline en entrenar el meta-controlador sobre cientos de sistemas robustamente estabilizables; luego, en línea, una nueva planta requiere solamente una inferencia de la red más algunas iteraciones del solver de proyección. La ventaja se manifiesta cuando el mismo template SDP se evalúa repetidamente con distintos parámetros, que es exactamente la situación en CPS con incertidumbre paramétrica."*

**Criterios cubiertos:** Diseño de solución (10%), Innovación (5%).

---

### **Slide 13 — Arquitectura LMI-Net (vista de bloques)**

**Contenido en pantalla — diagrama horizontal:**

```
[Sistema LPV: A_i, B_i]
        ↓ (24 features)
   ┌──────────────────────┐
   │   MLP backbone        │
   │  24 → 128 → 128 → 9   │
   │  Mish + LayerNorm     │
   └──────────────────────┘
        ↓ (ŷ: 9 params crudos)
   ╔══════════════════════════════════════════╗
   ║   CAPA LMI-Net                           ║
   ║   (Tang, Goertzen & Azizan, 2026)        ║
   ║                                          ║
   ║   Lifting:  C₁ (afín)  ∩  C₂ (cono PSD)  ║
   ║                                          ║
   ║   Forward:  Douglas–Rachford iterativo   ║
   ║   Backward: Algorithm Unrolling          ║
   ╚══════════════════════════════════════════╝
        ↓ (y*: 9 params factibles ✓)
   [Q, Y] → K = Y·Q⁻¹
        ↓
   Controlador estabilizador
```

*Cierre:* **"El MLP propone; la capa LMI-Net dispone. Toda salida cae, por construcción, en el conjunto factible."**

**Notas para el orador:**
> *"La arquitectura es modular en dos bloques. Arriba, un MLP estándar: tres capas densas con activación Mish y LayerNorm. Toma las 24 features que describen al sistema —los dos vértices del polítopo aplanados— y produce 9 parámetros crudos. Abajo, la capa LMI-Net: toma esos 9 parámetros, los proyecta sobre la intersección de un subespacio afín y el cono PSD, y devuelve 9 parámetros factibles. De esos 9 reconstruyo la matriz Q de Lyapunov simétrica y la ganancia Y. La ganancia final del controlador es K = Y · Q⁻¹."*

**Criterios cubiertos:** Diseño de solución (10%), Aprendizaje autónomo (10%).

---

### **Slide 14 — La capa LMI-Net en detalle: lifting y proyección**

**Contenido en pantalla:**

**Paso 1 — Lifting:** la LMI $F(y) \succeq 0$ se descompone introduciendo una variable matricial auxiliar $X$:
$$C_1 = \left\{(y, X) : F_0 + \textstyle\sum_i y_i F_i = X\right\}, \quad C_2 = \{(y, X) : X \succeq \epsilon I\}$$
$C_1$ es un **subespacio afín**; $C_2$ es el **cono PSD** desplazado.

**Paso 2 — Forward pass (Douglas–Rachford):** intercala dos proyecciones simples en cada iteración:
- $\Pi_{C_1}$: resolver un sistema lineal de mínimos cuadrados (pseudoinversa precomputable).
- $\Pi_{C_2}$: descomposición espectral + recorte de autovalores por debajo de $\epsilon$.

Combinación con paso de relajación: converge a un punto en $C_1 \cap C_2$.

**Paso 3 — Backward pass:** *algorithm unrolling* — cada iteración DR es una capa diferenciable; el gradiente fluye por la regla de la cadena a través del bucle desenrollado.

*Diagrama acompañante:* dos conjuntos en 2D ($C_1$ plano, $C_2$ cono) con una trayectoria zigzag desde $\hat y$ hasta $y^*$.

**Notas para el orador:**
> *"Acá está el corazón algorítmico. La idea del lifting, propuesta en el paper original, consiste en introducir una variable auxiliar X para descomponer la LMI en dos conjuntos sobre los cuales sí sé proyectar. El primero es un subespacio afín que codifica la estructura lineal de la LMI; proyectar sobre él es resolver un sistema de mínimos cuadrados, y la pseudoinversa se precomputa una vez. El segundo es el cono PSD desplazado por epsilon; proyectar sobre él es descomponer espectralmente y recortar los autovalores menores que epsilon. Douglas–Rachford intercala estas dos proyecciones con un paso de relajación. Y como cada paso es diferenciable, desenrollo todo el bucle como una secuencia de capas y dejo que la diferenciación automática haga el resto."*

**Criterios cubiertos:** Diseño de solución (10%), Aprendizaje autónomo (10%), Curiosidad científica (5%).

---

### **Slide 15 — Criterios de desempeño y entrenamiento auto-supervisado**

**Contenido en pantalla:**

**Paradigma:** **auto-supervisado** — la red no imita un solver; optimiza directamente la métrica física.

**Función de pérdida** (dos criterios implementados, ambos viables con la BD UNICAMP):

- **Criterio 1 — Tasa de decaimiento $\alpha$:** maximizar la velocidad de disipación de energía.
$$\mathcal{L}_\alpha = -\alpha \quad \text{sujeto a} \quad (A_i Q + B_i Y) + (A_i Q + B_i Y)^T + 2\alpha Q \prec 0$$

- **Criterio 2 — Volumen del elipsoide invariante:** minimizar la región de variación del estado bajo perturbaciones acotadas.
$$\mathcal{L}_{\text{vol}} = -\log\det(Q)$$

**Punto clave (resaltado):** la red no requiere etiquetas — el controlador $K_{\text{ref}}$ que viene en la BD se usa solo como certificado de factibilidad, **no como objetivo de imitación**.

**Notas para el orador:**
> *"Para entrenar la red, podría haber imitado los controladores K precalculados con punto interior que vienen en la base de datos. Pero eso solo replica un comportamiento que ya existe, y queda atrapado en sesgos del solver. En cambio, uso un esquema auto-supervisado: la red optimiza directamente el criterio físico de desempeño. Implementé dos criterios viables con los datos disponibles: maximizar la tasa de decaimiento alfa, que controla qué tan rápido disipa energía el sistema, y minimizar el volumen del elipsoide invariante, que controla qué tan acotada queda la trayectoria del estado. El K precalculado solo cumple un rol: garantizar que el problema es factible. Que existe al menos una solución."*

**Criterios cubiertos:** Diseño de solución (10%), Innovación (5%), Pensamiento crítico (5%).

---

## Bloque VI — Desarrollo y Resultados (3 slides)

### **Slide 16 — Metodología y dataset UNICAMP**

**Contenido en pantalla:**

- **Base de datos**: *Robust Analysis Routines and Database of Stabilizable Polytopic Systems* — Oliveira & Peres (UNICAMP).
- Estructura: arreglo MATLAB indexado por $(n_x, n_u, N, \text{instancia})$; 500 sistemas RS por configuración.
- **Subconjunto experimental (prueba de concepto):**
  - $n_x = 3$ estados, $n_u = 1$ actuador, $N = 2$ vértices.
  - Split: 400 train / 100 test.
- **Pipeline (4 etapas):** lectura `.mat` → aplanado a vector de 24 features → forward + DR → backprop con unrolling.
- **Hiperparámetros**: Adam, lr = $10^{-3}$, 50 épocas, batch 16, 30 iter DR (train) / 5000 iter DR (inferencia), FP64.

**Notas para el orador:**
> *"Para que los resultados sean comparables con la literatura, uso la base de datos de UNICAMP del grupo de Oliveira y Peres, que es el benchmark estándar en control robusto LMI. Para esta primera prueba de concepto fijé la topología en tres estados, un actuador y dos vértices —400 sistemas para entrenar, 100 para validar—. Esa elección no es arbitraria: dos vértices representan geométricamente un segmento de recta en el espacio paramétrico, la forma más limpia de incertidumbre, y mantener la dimensionalidad fija evita técnicas de zero-padding que dañarían las propiedades PSD."*

**Criterios cubiertos:** Desarrollo (5%), Aprendizaje autónomo (10%).

---

### **Slide 17 — Resultados preliminares: estabilización**

**Contenido en pantalla — tabla limpia:**

| Métrica | Resultado |
|---|---|
| Sistemas estrictamente estables ($\text{Re}(\lambda_{\max}) < 0$) | **93.00 %** |
| Cumplimiento de tasa de decaimiento ($\leq -0.01$) | **88.00 %** |
| Promedio del peor autovalor en lazo cerrado | $-0.0743$ |
| Configuración | LMI-Net (Mish + LayerNorm), FP64, 5000 iter DR |
| Conjunto de prueba | 100 sistemas nunca vistos |

*Mini-gráfico opcional:* trayectorias temporales de un sistema controlado por LMI-Net (estable) vs. baseline soft-constrained (inestable).

**Notas para el orador:**
> *"Sobre el test set de 100 sistemas nunca vistos durante el entrenamiento, el 93% resultó estrictamente estable: el peor autovalor en lazo cerrado quedó en el semiplano izquierdo. El 88% además cumplió con el margen exigido de alfa igual a 0.01. El promedio del peor autovalor fue de -0.0743, un margen muy seguro. La brecha del 7% al 100% no es un fallo teórico: es atribuible al gap entre las 30 iteraciones DR del entrenamiento y las 5000 de la inferencia, una limitación de memoria que abordaré en el análisis de riesgos."*

**Criterios cubiertos:** Desarrollo (5%), Aprendizaje autónomo (10%), Pensamiento crítico (5%).

---

### **Slide 18 — Resultados preliminares: velocidad de inferencia**

**Contenido en pantalla — gráfico de barras horizontal (ms/muestra):**

- LMI-Net (DR 500 iter):   **0.21 ms**
- LMI-Net (DR 1000 iter):  **0.41 ms**
- LMI-Net (DR 2000 iter):  **0.83 ms**
- LMI-Net (DR 4000 iter):  **1.64 ms**
- **CVXPY/SCS:             4.29 ms** ← referencia

*Cierre:* **"Entre 2.6× y 20× más rápido que el solver clásico, manteniendo cero violaciones en la región interior del dataset."**

> *Nota:* los datos provienen de los benchmarks del paper original (Tang et al., 2026). Mi pipeline reproduce el rango esperado en el sub-conjunto UNICAMP correspondiente.

**Notas para el orador:**
> *"Y el otro lado de la moneda: la velocidad. Con 2000 iteraciones DR LMI-Net no presenta violaciones en el conjunto de entrenamiento, y a la vez es aproximadamente 5 veces más rápido que CVXPY con el solver SCS. Si me permito más iteraciones —digamos 4000— sigo siendo más rápido que el solver clásico. Esto valida la hipótesis central: las restricciones duras son compatibles con una inferencia significativamente más eficiente que la del enfoque tradicional. Importante: estos números son del paper original; mis resultados sobre UNICAMP reproducen ese rango."*

**Criterios cubiertos:** Desarrollo (5%), Análisis comparativo (10%), Innovación (5%).

---

## Bloque VII — Análisis del Proyecto (5 slides)

### **Slide 19 — Línea de trabajo en curso: rigidez dimensional**

**Contenido en pantalla — diagrama "problema vs. caminos":**

**El problema:**
- El MLP backbone tiene dimensión de entrada **fija** en 24 features (definida por $N=2$, $n_x=3$, $n_u=1$).
- Si cambia *cualquiera* de esos parámetros topológicos, **toda la red debe redimensionarse y reentrenarse**.
- Esto contradice parcialmente la idea de un "meta-controlador" genérico.

**Caminos en exploración:**
1. **Padding canónico**: rellenar con ceros hasta una dimensión máxima — descarta propiedades PSD, rechazado.
2. **Embedding por matriz estructurada**: mapear cualquier sistema a un espacio de representación común antes del MLP.
3. **Arquitecturas equivariantes**: Graph Neural Networks (los vértices del polítopo como nodos) o Transformers (atención sobre vértices).
4. **Hipernetworks**: una red genera los pesos de otra red especializada en cada topología.

**Estado actual:** evaluando opciones 3 y 4 como línea de trabajo para los últimos meses de la memoria y, probablemente, como puente al magíster.

**Notas para el orador:**
> *"Quiero ser totalmente transparente sobre una limitación que estoy trabajando activamente. La arquitectura actual tiene una dimensión de entrada fija: 24 features, calculada para 2 vértices, 3 estados y 1 actuador. Si cambia cualquier dimensión topológica del sistema, necesito reentrenar una red distinta. Esto debilita la promesa de un meta-controlador genérico. Estoy evaluando cuatro caminos: padding canónico, que descarté porque destruye las propiedades matriciales; embedding estructurado; arquitecturas equivariantes como Graph Neural Networks o Transformers, donde cada vértice del polítopo sería un nodo o un token; y por último hipernetworks. Esta es una línea activa de trabajo y probablemente un puente natural hacia mi magíster en ciencias de la computación."*

**Criterios cubiertos:** Pensamiento crítico (5%), Innovación (5%), Aprendizaje autónomo (10%).

---

### **Slide 20 — Análisis de riesgos**

**Contenido en pantalla — tabla P × I × Mitigación:**

| Riesgo | P | I | Mitigación |
|---|---|---|---|
| **R1. Gap de entrenamiento** (30 vs 5000 iter DR) | Alta | Alto | Explorar diferenciación implícita estilo DEQ o Deep Declarative Networks |
| **R2. Inestabilidad numérica en FP32** | Alta | Alto | Forzar FP64; ya implementado |
| **R3. Rigidez arquitectónica** | Alta | Medio | Línea de trabajo dedicada (Slide 19) |
| **R4. Sensibilidad al margen $\epsilon$** del cono PSD | Media | Alto | Calendarización de $\epsilon$ durante entrenamiento |
| **R5. Casos marginales** (polos cerca del eje imaginario) | Media | Medio | Aumentar iteraciones DR en inferencia para esos casos |
| **R6. No generalización a OOD** | Media | Medio | Evaluación con OOD-SLOW / OOD-LARGE del paper original |

**Notas para el orador:**
> *"Los riesgos no son hipotéticos: ya los he encontrado en práctica. El más importante es el 'gap de entrenamiento': entreno con 30 iteraciones de Douglas-Rachford porque más explotaría la memoria GPU al desenrollar el grafo, pero evalúo con 5000 iteraciones para garantizar factibilidad estricta. Eso crea una discrepancia entre lo que la red optimiza y lo que el solver finalmente produce. Mi camino principal de mitigación es la diferenciación implícita, que calcula gradientes analíticamente en el punto fijo sin necesidad de almacenar el grafo entero."*

**Criterios cubiertos:** Análisis de riesgos (5%), Pensamiento crítico (5%).

---

### **Slide 21 — Restricciones del proyecto**

**Contenido en pantalla — cuatro categorías:**

- **Restricciones técnicas:**
  - Dimensionalidad topológica fija ($n_x=3$, $n_u=1$, $N=2$).
  - Precisión computacional FP64 obligatoria (sacrifica throughput).
  - Memoria GPU limita el desenrollado a ~30 iteraciones DR durante entrenamiento.

- **Restricciones de datos:**
  - Solo sistemas robustamente estabilizables (RS) — no se valida sobre sistemas no-RS por diseño.
  - 500 sistemas por configuración topológica.
  - La BD **no incluye matriz de canal de perturbación** $B_w$: por eso H∞ queda fuera del alcance actual.

- **Restricciones metodológicas:**
  - Solo realimentación de estados (no realimentación estática de salida, que es BMI no convexa).
  - Lyapunov cuadrático común (no parámetro-dependiente).

- **Restricciones de validación:**
  - Validación exclusivamente en simulación; no hay integración con planta física.

**Notas para el orador:**
> *"Toda metodología vive bajo restricciones. Las mías son explícitas: dimensionalidad fija, precisión doble obligatoria, memoria que limita el unrolling, y foco exclusivo en sistemas que ya son matemáticamente estabilizables. Un punto importante: la base de datos UNICAMP contiene las matrices Ai y Bi del polítopo y el controlador K precalculado, pero no contiene una matriz de canal de perturbación Bw. Eso descarta de plano el criterio H-infinito, que requiere caracterizar un canal de disturbio. Es una restricción de los datos, no del método."*

**Criterios cubiertos:** Restricciones (5%), Pensamiento crítico (5%).

---

### **Slide 22 — Condiciones para operación adecuada (técnica + gestión del cambio)**

**Contenido en pantalla — dos columnas:**

**Aspectos técnicos:**
- Hardware: GPU CUDA con ≥8 GB VRAM (entrenamiento); CPU/GPU genérica para inferencia.
- Stack: Python 3.10+, PyTorch ≥ 2.x, precisión doble habilitada.
- Pipeline de validación obligatorio: cada modelo entrenado pasa por suite de ≥100 sistemas test antes de despliegue.
- Monitoreo continuo: registrar el peor autovalor en cada inferencia para detectar drift.

**Implantación organizacional y gestión del cambio:**
- **Capacitación**: el equipo de control debe entender que el meta-controlador **complementa**, no reemplaza, al solver clásico. El solver clásico sigue siendo el oráculo en simulación.
- **Documentación viva**: cada modelo entrenado se versiona con hash del dataset, hiperparámetros, métricas y *model card*.
- **Política de fallback**: si la métrica de violación supera un umbral en runtime, conmutar a controlador clásico precalculado o detener actuador.
- **Mantención**: reentrenar trimestralmente o ante cambios documentados en la planta (envejecimiento, reparaciones).

**Notas para el orador:**
> *"Aunque la tesis es de naturaleza teórica y experimental, pensé deliberadamente qué implicaría llevar esto a producción. Lo más importante en términos de gestión del cambio es que el meta-controlador no reemplaza al ingeniero de control: lo complementa. El solver clásico sigue siendo el ground truth en simulación, y la red es la actuación en runtime. Y como cualquier sistema de aprendizaje automático en producción, requiere versionado, monitoreo y política de fallback. Esto no es trivial: es lo que separa una prueba de concepto de un sistema desplegable."*

**Criterios cubiertos:** Condiciones de operación (10%), Innovación (5%).

---

### **Slide 23 — Costos y beneficios**

**Contenido en pantalla — tabla balanceada:**

| | **Costos** | **Beneficios** |
|---|---|---|
| **Computacionales** | Entrenamiento offline costoso (~horas en GPU); FP64 obligatoria (≈50% throughput menor que FP32) | Inferencia 2.6×–20× más rápida que solver clásico; tiempo casi constante por instancia |
| **De ingeniería** | Mayor complejidad de implementación; pipeline ML completo (datos + modelo + monitoreo) | Arquitectura entrenada una vez, desplegada miles de veces |
| **Académicos / científicos** | Aún no hay generalización a topologías arbitrarias; gap teoría–práctica en el unrolling | Aporta una pieza al puente control–DL; arquitectura reusable para otras LMIs (filtrado, observadores) |
| **Operacionales** | Curva de aprendizaje del equipo (capacitación); dependencia de hardware GPU | Habilita aplicaciones donde el solver clásico es inviable (microgrids, vehículos autónomos) |

*Cierre (resaltado):* **"El costo se paga una vez offline; el beneficio se cobra en cada inferencia online."**

**Notas para el orador:**
> *"Los costos no son solo económicos: son cognitivos, organizacionales y técnicos. Pero el balance favorece claramente al enfoque: pagar un costo grande una vez en entrenamiento offline para obtener un beneficio recurrente en cada decisión de control. Y para clases de problemas donde el solver clásico era simplemente inviable —microgrids con dinámica rápida, vehículos autónomos en bordes embebidos sin solver instalable—, este enfoque habilita aplicaciones nuevas."*

**Criterios cubiertos:** Costos y beneficios (10%), Pensamiento crítico (5%).

---

## Bloque VIII — Cierre (2 slides)

### **Slide 24 — Conclusiones del avance y trabajo futuro**

**Contenido en pantalla — dos columnas:**

**Conclusiones del avance:**
- ✅ Pipeline reproducible UNICAMP → PyTorch implementado.
- ✅ Arquitectura LMI-Net adaptada al problema politópico LPV.
- ✅ 93% de estabilización y 88% de cumplimiento de tasa de decaimiento sobre test set.
- ✅ Limitaciones identificadas y caracterizadas cuantitativamente.

**Trabajo futuro (en orden de prioridad):**
1. **Explorar algoritmos alternativos a Douglas–Rachford** para la capa de proyección (por ejemplo, **Dykstra**, que converge exactamente a la proyección euclidiana sobre la intersección; **ADMM** con descomposiciones modificadas; **operator splitting** con aceleración).
2. **Diferenciación implícita** en el pase hacia atrás (estilo DEQ / Deep Declarative Networks): cerrar el gap de entrenamiento sin agotar memoria.
3. **Nuevos criterios de desempeño viables con la BD UNICAMP**:
   - **Costo cuadrático garantizado tipo $\mathcal{H}_2$/LQ** (pesos de diseño $Q_x \succeq 0$, $R \succ 0$).
   - **Condicionamiento del elipsoide $\kappa(Q)$** — depende solo de $Q$.
   - **Margen de factibilidad robusta** — maximizar $t$ tal que $F_i(y) \succeq tI$.
   - **Aprovechamiento de $K_{\text{ref}}$**: línea base de mejora relativa, o regularizador de imitación opcional.
4. **Abordar la rigidez dimensional** (Slide 19): Graph Neural Networks, Transformers, o hipernetworks.
5. **Continuación en Magíster en Ciencias de la Computación**: aplicaciones a control en red bajo ciberataques (FDI / DoS).

**Notas para el orador:**
> *"El avance demuestra que la arquitectura funciona empíricamente, que las limitaciones están bien caracterizadas, y que hay una hoja de ruta clara. El trabajo futuro no es una lista de deseos: es un orden de prioridades técnicas. Notar que descarté el criterio H-infinito porque requiere una matriz de canal de perturbación que la base de datos no contiene; los criterios listados son todos viables con los datos disponibles. Y, lo más relevante para mi trayectoria, este trabajo abre las preguntas que abordaré durante mi magíster."*

**Criterios cubiertos:** Conclusiones (5%), Pensamiento crítico (5%), Curiosidad científica (5%).

---

### **Slide 25 — Cierre**

**Contenido en pantalla:**
- *"Gracias. Quedo abierto a preguntas y comentarios."*
- Datos de contacto.
- (Opcional) QR al repositorio del código.

**Notas para el orador:**
> *"Gracias por su atención. Quedo abierto a sus preguntas y comentarios."*

**Criterios cubiertos:** Respuesta a comisión (5%) — se demuestra en vivo.

---

## Anexo A — Slides de respaldo (NO se muestran; se invocan ante preguntas)

- **A.1 — Ecuaciones detalladas de Douglas–Rachford** (proximales, paso de relajación, complejidad por iteración).
- **A.2 — Justificación de Mish vs ReLU y LayerNorm** (suavidad de gradientes vs. sensibilidad del solver).
- **A.3 — Resultados extendidos sobre OOD** (OOD-SLOW: violación 3.4% con DR=4000; OOD-LARGE).
- **A.4 — Comparativa con baseline *soft-constrained*** (79.2% de infactibilidad en OOD).
- **A.5 — Análisis de sensibilidad al margen $\epsilon$**.
- **A.6 — Hiperparámetros completos, seed, configuración de PyTorch**.

---

## Anexo B — Preguntas anticipables y respuestas modelo

Útil para preparar — NO es slide.

**Para el evaluador de matemáticas / Deep Learning:**

1. **"¿Por qué Mish y no ReLU? ¿No es más estándar?"**
   → Mish es C∞ y permite gradientes negativos no cero. Douglas–Rachford involucra descomposiciones espectrales que son numéricamente sensibles a discontinuidades en la derivada. ReLU introduce un quiebre en cero que se traduce en ruido en los gradientes que fluyen por el unrolling.

2. **"¿Cómo garantizan que el unrolling con 30 iteraciones converge a algo significativo en entrenamiento?"**
   → No converge a la proyección exacta; converge a un punto "casi factible" del cual el solver de inferencia con 5000 iteraciones parte. Es lo que el paper llama "warm-start interno". El gap es justamente la limitación principal documentada.

3. **"¿Por qué auto-supervisado y no supervisado con los $K_{\text{ref}}$ de la BD?"**
   → Imitar al solver clásico hereda sus sesgos (típicamente conservador). Auto-supervisado permite explorar el interior del dominio factible, no solo la solución que produce el solver. Los $K_{\text{ref}}$ pueden usarse como regularizador opcional —es uno de mis criterios de trabajo futuro—.

4. **"¿La capa LMI-Net es estrictamente la proyección euclidiana?"**
   → No: con el paso de relajación de Douglas–Rachford, el resultado es *algún* punto en $C_1 \cap C_2$, no necesariamente el más cercano a $\hat y$. Esa es precisamente la razón por la que Dykstra figura en mi trabajo futuro: Dykstra sí converge a la proyección euclidiana exacta.

**Para el evaluador de algoritmos / optimización:**

5. **"¿Por qué Douglas–Rachford y no ADMM, Dykstra u otro splitting?"**
   → DR es el más simple para esta descomposición de dos conjuntos y el que usa el paper original. ADMM es equivalente bajo ciertas condiciones de dualidad. Dykstra es matemáticamente más fiel al objetivo de proyección, pero el paper original eligió DR; mi trabajo futuro plantea probar Dykstra y comparar empíricamente.

6. **"¿Cuál es la complejidad por iteración de la capa?"**
   → Por iteración DR: $\mathcal{O}(n^3)$ por la descomposición espectral en $\Pi_{C_2}$ + $\mathcal{O}(m^2)$ por el sistema lineal de $\Pi_{C_1}$ (con pseudoinversa precomputada, una sola vez). Para $n=3$, $m=9$, esto es trivial; el cuello de botella aparece en órdenes grandes.

7. **"¿Qué pasa si $C_1 \cap C_2 = \emptyset$?"**
   → DR no converge (oscila). Por eso la BD UNICAMP es esencial: solo contiene sistemas robustamente estabilizables, lo que garantiza que la intersección es no vacía.

8. **"¿Cómo se compara con MPC?"**
   → MPC resuelve un QP en cada paso temporal; LMI-Net resuelve un SDP/LMI offline una vez y aprende el mapping. No son sustitutos: MPC maneja restricciones de estado/entrada, LMI-Net maneja garantías de estabilidad. Pueden combinarse.

**Generales:**

9. **"¿Qué hace que su memoria sea diferente del paper de Tang et al.?"**
   → Tang et al. usan un benchmark genérico de sistemas perturbados; yo aplico LMI-Net específicamente al problema politópico LPV sobre la base UNICAMP, que es el benchmark estándar en control robusto. Además, propongo extensiones concretas (criterios de desempeño viables, alternativas a DR) y caracterizo limitaciones específicas (rigidez dimensional, gap de entrenamiento) que el paper original no aborda en profundidad.

10. **"¿Por qué solo $N=2$ y $n_x=3$?"**
    → Prueba de concepto controlada. Cualquier dimensión es teóricamente válida; el límite actual es la rigidez arquitectónica (Slide 19), que es justamente una línea de trabajo activa.

---

## Recomendaciones finales antes de pasarlo a Claude Design

1. **Estilo visual:** academic-minimal. Tipografía: sans-serif para títulos (Inter, IBM Plex Sans), serif para ecuaciones (mejor renderizado LaTeX). Evita degradados y "decoraciones".

2. **Iconografía:** usa íconos vectoriales solo donde aporten (Slide 2 para CPS; Slide 12 para offline/online). El resto, evítalos.

3. **Ecuaciones:** renderizadas como LaTeX/MathJax, nunca como capturas de pantalla.

4. **Diagramas críticos** (invertir tiempo): Slide 12 (offline-online), Slide 13 (arquitectura), Slide 14 (proyección geométrica), Slide 18 (gráfico de barras).

5. **Colores semánticos consistentes:** verde para resultados positivos, ámbar/rojo para riesgos/limitaciones, azul para conceptos neutros, gris para tablas/baselines.

6. **Tiempo por slide:** 45–60 s en promedio. Slides 13, 14, 17 y 18 admiten 75–90 s; los demás más cortos.

7. **Ensayos:** mínimo 3 ensayos en voz alta cronometrados. Apunta a 18–20 min totales, dejando margen.

8. **Importante para Claude Design:** no pegues el markdown completo en un solo prompt. Primero pide el theme con los slides 1, 13 y 17 como referencia. Cuando apruebes el estilo, pasa el resto en bloques de 4–5 slides.
