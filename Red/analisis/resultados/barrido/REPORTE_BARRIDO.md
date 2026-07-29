# Reporte del barrido LMI-Net

_Generado: 2026-07-29 01:44:15_

## Cobertura

| etapa | completas | esperadas | % |
|---|---|---|---|
| E1 | 240 | 240 | 100% |
| E2 | 48 | 48 | 100% |
| E3 | 36 | 36 | 100% |
| E4 | 119 | 120 | 99% |


Filas crudas analizadas: **27,742,400** (una por sistema × anillo × época × dr_eval). Métrica de referencia: **% estabilizado en A2** (época 400, dr_eval=8000), promedio sobre semillas.

## 🏆 Mejor configuración (generalización A2)

A la época 400, dr_eval=8000, la mejor combinación en el anillo estructural A2 es:

- **pérdida = `margen_norm`**, arquitectura = `actuadores`, backprop = `unrolling` (dr_train=30), α = 0.01, n_x = 3, train_size = 400

- **A2: 97.4% estabilizado**, 90.9% cumple decay; iters_min mediano = 100, κ(Q) mediano = 1.67 (sobre 2 semillas)


## Ranking global de configuraciones

### Anillo A2 — estructural / zero-shot (época 400, dr_eval=8000)

| pérdida | arqu. | backprop | dr_tr | α | n_x | sz | A2 %estab | %decay | iters_min | κ(Q) | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| margen_norm | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 97.4 | 90.9 | 100 | 1.67 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.01 | 3 | 150 | 97.2 | 91.3 | 100 | 7.38 | 2 |
| control_margen | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 96.9 | 89.8 | 100 | 1 | 2 |
| control | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 96.9 | 89.9 | 100 | 1 | 2 |
| margen_norm | actuadores | unrolling | 30 | 0.01 | 3 | 150 | 96.8 | 90.1 | 100 | 2.94 | 2 |
| control_margen | actuadores | unrolling | 5 | 0.01 | 3 | 150 | 96.6 | 90.5 | 100 | 5.24 | 2 |
| control_margen | actuadores | unrolling | 30 | 0.01 | 3 | 150 | 96.5 | 89.2 | 100 | 1.83 | 2 |
| margen_norm | actuadores | unrolling | 30 | 0.01 | 4 | 150 | 96.5 | 88.2 | 100 | 3.89 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 96.2 | 94.9 | 100 | 11.6 | 2 |
| control_margen | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 96.2 | 95.0 | 100 | 3.28 | 2 |
| margen_norm | actuadores | unrolling | 120 | 0.01 | 3 | 150 | 96.1 | 88.6 | 100 | 1.91 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.05 | 3 | 150 | 96.0 | 66.6 | 100 | 5.39 | 2 |
| esfuerzo | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 95.7 | 95.0 | 100 | 25.3 | 2 |
| condicionamiento | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 95.7 | 88.4 | 100 | 1.21 | 2 |
| control_margen | actuadores | unrolling | 5 | 0.05 | 3 | 150 | 95.6 | 65.4 | 100 | 2.73 | 2 |


### Anillo A1 — visto (época 400, dr_eval=8000)

| pérdida | arqu. | backprop | dr_tr | α | n_x | sz | A1 %estab | %decay | iters_min | κ(Q) | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| esfuerzo | vanilla | unrolling | 30 | 0.01 | 2 | 150 | 97.0 | 87.5 | 100 | 5.18 | 2 |
| control | vanilla | unrolling | 30 | 0.01 | 2 | 150 | 97.0 | 87.0 | 100 | 1.07 | 2 |
| condicionamiento | vanilla | unrolling | 30 | 0.01 | 2 | 150 | 97.0 | 85.5 | 100 | 1.01 | 2 |
| control_margen | vanilla | unrolling | 30 | 0.01 | 2 | 150 | 96.5 | 89.0 | 100 | 1 | 2 |
| paper | vanilla | unrolling | 30 | 0.01 | 2 | 150 | 96.5 | 87.5 | 100 | 1.16 | 2 |
| control_margen | actuadores | unrolling | 30 | 0.01 | 2 | 150 | 95.7 | 86.2 | 100 | 1.19 | 2 |
| control | vertices | unrolling | 30 | 0.01 | 2 | 150 | 95.7 | 84.7 | 100 | 1 | 2 |
| condicionamiento | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 95.5 | 89.5 | 100 | 1.69 | 2 |
| control | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 95.5 | 89.0 | 100 | 2.06 | 2 |
| control_margen | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 95.0 | 90.0 | 100 | 1.48 | 2 |
| esfuerzo | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 95.0 | 90.0 | 100 | 19.1 | 2 |
| control | actuadores | unrolling | 30 | 0.01 | 2 | 150 | 95.0 | 84.3 | 100 | 1.19 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.01 | 3 | 150 | 94.7 | 85.5 | 100 | 6.79 | 2 |
| paper | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 94.5 | 88.0 | 100 | 1.9 | 2 |
| margen_norm | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 94.5 | 87.5 | 100 | 5.77 | 2 |


## Efecto de cada eje (A2, marginando el resto)


**Función de pérdida** — mejor: `condicionamiento` (89.2% A2)

| Función de pérdida | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| condicionamiento | 89.2 | 71.8 | 250 | 145000 |
| margen_norm | 87.4 | 69.4 | 100 | 142500 |
| control_margen | 85.7 | 67.1 | 250 | 145000 |
| esfuerzo | 83.2 | 70.9 | 100 | 145000 |
| paper | 82.2 | 63.7 | 250 | 145000 |
| control | 78.0 | 59.6 | 250 | 145000 |


**Backpropagation** — mejor: `unrolling` (89.5% A2)

| Backpropagation | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| unrolling | 89.5 | 72.0 | 250 | 720000 |
| implicit | 59.0 | 42.9 | 250 | 147500 |


**Arquitectura** — mejor: `vertices` (89.9% A2)

| Arquitectura | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| vertices | 89.9 | 75.7 | 100 | 24000 |
| actuadores | 84.1 | 66.8 | 250 | 843500 |


**α (decay prescrito)** — mejor: `0.01` (87.4% A2)

| α (decay prescrito) | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| 0.01 | 87.4 | 79.6 | 100 | 417500 |
| 0.05 | 81.8 | 52.7 | 250 | 150000 |
| 0.001 | 81.8 | 80.6 | 250 | 150000 |
| 0.1 | 80.5 | 33.1 | 250 | 150000 |


**Presupuesto DR de entrenamiento** — mejor: `30` (93.3% A2)

| Presupuesto DR de entrenamiento | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| 30 | 93.3 | 80.7 | 100 | 360000 |
| 5 | 92.0 | 71.0 | 100 | 120000 |
| 120 | 85.5 | 63.1 | 250 | 120000 |
| 500 | 79.4 | 56.0 | 500 | 120000 |
| — | 59.0 | 42.9 | 250 | 147500 |


**Orden del sistema n_x** — mejor: `4` (93.6% A2)

| Orden del sistema n_x | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| 4 | 93.6 | 84.6 | 100 | 36000 |
| 2 | 91.2 | 78.8 | 100 | 12000 |
| 5 | 90.3 | 79.2 | 250 | 36000 |
| 3 | 83.5 | 65.5 | 250 | 783500 |


**Tamaño de entrenamiento** — mejor: `400` (95.6% A2)

| Tamaño de entrenamiento | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| 400 | 95.6 | 88.9 | 100 | 30000 |
| 50 | 93.4 | 86.4 | 100 | 30000 |
| 150 | 83.5 | 65.6 | 250 | 807500 |


## Eficiencia (menor iters_min con A2 ≥ 90% estabilizado)

| pérdida | arqu. | backprop | dr_tr | α | n_x | sz | A2 %estab | %decay | iters_min | κ(Q) | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| margen_norm | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 97.4 | 90.9 | 100 | 1.67 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.01 | 3 | 150 | 97.2 | 91.3 | 100 | 7.38 | 2 |
| control_margen | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 96.9 | 89.8 | 100 | 1 | 2 |
| control | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 96.9 | 89.9 | 100 | 1 | 2 |
| margen_norm | actuadores | unrolling | 30 | 0.01 | 3 | 150 | 96.8 | 90.1 | 100 | 2.94 | 2 |
| control_margen | actuadores | unrolling | 5 | 0.01 | 3 | 150 | 96.6 | 90.5 | 100 | 5.24 | 2 |
| control_margen | actuadores | unrolling | 30 | 0.01 | 3 | 150 | 96.5 | 89.2 | 100 | 1.83 | 2 |
| margen_norm | actuadores | unrolling | 30 | 0.01 | 4 | 150 | 96.5 | 88.2 | 100 | 3.89 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 96.2 | 94.9 | 100 | 11.6 | 2 |
| control_margen | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 96.2 | 95.0 | 100 | 3.28 | 2 |


## Duelo de arquitecturas × n_x

Config base controlada (`unrolling / dr30 / α=0.01 / sz150`); % estabilizado promedio sobre las 6 pérdidas × 2 semillas. Única diferencia entre celdas: arquitectura y orden. `vanilla` no es invariante → no tiene A2.


**A2 — generalización zero-shot (topología N=5 / n_u=2 no vista)**

| n_x | actuadores | vertices |
|---|---|---|
| 2 | 91.4 | 91.1 |
| 3 | 95.2 | 90.7 |
| 4 | 94.1 | 91.0 |
| 5 | 91.1 | 86.7 |


**A1 — sistemas vistos**

| n_x | actuadores | vertices | vanilla |
|---|---|---|---|
| 2 | 91.2 | 91.5 | 95.8 |
| 3 | 91.2 | 91.3 | 95.0 |
| 4 | 88.4 | 88.4 | 92.5 |
| 5 | 87.4 | 87.5 | 91.8 |


_Lectura: en A2, `actuadores` gana en todos los órdenes (su ventaja sobre `vertices` crece con n_x); en A1, `vanilla` domina pero no generaliza. Cobertura: `actuadores` de E1/E2, `vertices`/`vanilla` de E4._


---
_Métricas por-sistema promediadas en el análisis. A2 = topologías retenidas (N=5, n_u=2) nunca vistas en entrenamiento._
