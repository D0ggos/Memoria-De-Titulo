# Contexto: base de datos `DB_ssf_RS_500_c.mat`

Resumen denso para dar contexto a un chat sin cargar el `.txt` (21 MB).
Derivado empíricamente de los 14 000 sistemas (ver `analisis_db.tex` y `figuras/`).

## Qué es
Sistemas lineales **politópicos de tiempo continuo** + una ganancia de
realimentación de estado estática (`ssf`) que los estabiliza de forma robusta.
`RS` = robustamente (cuadráticamente) estabilizable.

Sistema incierto = envolvente convexa de `N` vértices:
`ẋ = A(θ)x + B(θ)u`, con `(A(θ),B(θ)) = Σ θ_i (A_i,B_i)`, `θ_i≥0`, `Σθ_i=1`.
Control: `u = Kx`, con **una sola** `K` común a todo el politopo.

## Estructura del `.mat`
- `BASE[n_x, n_u, N, caso]` — arreglo de celdas, forma nominal `5×2×5×500`.
- Cada celda poblada: campos `A` (celda de N matrices `A_i ∈ ℝ^{n_x×n_x}`),
  `B` (celda de N matrices `B_i ∈ ℝ^{n_x×n_u}`), `K ∈ ℝ^{n_u×n_x}`.
- Metadatos: `cases=500`, `order_range`, `inputs_range`, `vertices_range`.

## Dimensiones y cobertura
- `n_x ∈ {2,3,4,5}` (orden), `n_u ∈ {1,2}` (actuadores), `N ∈ {2,3,4,5}` (vértices).
- Soporte: `n_u=1 ⇒ n_x∈{2..5}`; `n_u=2 ⇒ n_x∈{3..5}`; siempre `N≥2`.
- **Diseño factorial completo y balanceado**: 28 celdas válidas × 500 = **14 000 sistemas**
  (8 000 con `n_u=1`, 6 000 con `n_u=2`). 0 faltantes, 0 duplicados.

## Hechos clave (verificados recomputando autovalores)
- **Planta difícil**: 98.8 % inestable en lazo abierto; abscisa espectral del
  peor vértice con mediana 30.9 y máx ~832. Solo 1.2 % estable sin control.
- **Etiqueta válida al 100 %**: la `K` almacenada deja todos los vértices
  Hurwitz (`A_i+B_iK`) en los 14 000 sistemas. 100 % controlables.
- **Estabilización al borde (hallazgo principal)**: la tasa de decaimiento
  lograda `α = -max_i max Re λ(A_i+B_iK)` es casi marginal — mediana
  `α≈6.3e-4`, y **84.9 % con α<1e-2**. Distribución bimodal (cola minoritaria
  con α hasta 10.6). El polo dominante queda sobre el eje; el resto del espectro
  se empuja muy a la izquierda (mediana de Re sobre todos los λ de lazo cerrado = -21.4).
  ⇒ La `K` parece construida por **factibilidad** de la LMI, no por maximizar el decay.
- **Esfuerzo `‖K‖₂`**: mediana 17.4, máx 98. Correlaciona con `‖A‖` (ρ=0.90),
  dispersión del politopo (0.85) e inestabilidad OL (0.63).
- **`α` no correlaciona con nada** (|ρ|≤0.04): el margen marginal es decisión de
  diseño, no dificultad del sistema.
- **Rango dinámico grande**: `max‖A_i‖₂` de 12 a 1012 (mediana 131); `‖B_i‖₂` en
  [0.74, 16.8]. Condicionamiento mediana 53, cola del 0.39 % con κ>1e4 (máx 5.7e5).
- **Geometría**: dispersión del politopo `max_i‖A_i-Ā‖₂` (mediana 97.5) crece con N
  ⇒ más vértices = más incertidumbre. Conviene evaluar estratificado por N.

## Normalización en la ingesta (`pipeline/data_loader.py`)
Por sistema se divide A y B por `γ = max(max|A|, max|B|)` → entradas en [-1,1].
Preserva el signo de los autovalores (no cambia estabilizabilidad), escala el
decay por 1/γ. `K` se devuelve **sin** normalizar.

## Notas de procedencia (POR CONFIRMAR con el autor)
- Procedimiento de muestreo de los vértices `(A_i,B_i)`: desconocido.
- Cómo se calculó `K`: consistente con LMI de estabilizabilidad cuadrática
  (certificado común `(Q,Y)`, `K=YQ⁻¹`) sin maximizar α — pero sin confirmar.
- `c` en `_500_c` / `_100_c`: sin confirmar.
