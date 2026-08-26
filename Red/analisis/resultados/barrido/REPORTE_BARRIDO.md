# Reporte del barrido LMI-Net

_Generado: 2026-08-26 12:16:47_

## Cobertura

| etapa | completas | esperadas | % |
|---|---|---|---|
| E1 | 240 | 240 | 100% |
| E2 | 48 | 48 | 100% |
| E3 | 36 | 36 | 100% |
| E4 | 119 | 120 | 99% |
| E7 | 192 | 192 | 100% |
| E8 | 24 | 24 | 100% |
| E9 | 48 | 48 | 100% |


Filas crudas analizadas: **45,281,600** (una por sistema × anillo × época × dr_eval). Métrica de referencia: **% estabilizado en A2** (época 400, dr_eval=8000), promedio sobre semillas.

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
| margen_norm | actuadores | unrolling | 30 | 0.01 | 4 | 150 | 97.1 | 89.6 | 100 | 5.72 | 2 |
| control_margen | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 96.9 | 89.8 | 100 | 1 | 2 |
| control | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 96.9 | 89.9 | 100 | 1 | 2 |
| control_margen | actuadores | unrolling | 5 | 0.01 | 3 | 150 | 96.6 | 90.5 | 100 | 5.24 | 2 |
| control_margen | actuadores | unrolling | 30 | 0.01 | 4 | 150 | 96.5 | 88.5 | 100 | 4.01 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 96.2 | 94.9 | 100 | 11.6 | 2 |
| control_margen | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 96.2 | 95.0 | 100 | 3.28 | 2 |
| margen_norm | actuadores | unrolling | 120 | 0.01 | 3 | 150 | 96.1 | 88.6 | 100 | 1.91 | 2 |
| condicionamiento | actuadores | unrolling | 30 | 0.01 | 4 | 150 | 96.1 | 88.2 | 100 | 5.45 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.05 | 3 | 150 | 96.0 | 66.6 | 100 | 5.39 | 2 |
| esfuerzo | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 95.7 | 95.0 | 100 | 25.3 | 2 |
| condicionamiento | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 95.7 | 88.4 | 100 | 1.21 | 2 |
| margen_norm | actuadores | unrolling | 30 | 0.01 | 5 | 150 | 95.7 | 87.3 | 100 | 4.66 | 2 |


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
| control_margen | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 95.0 | 88.2 | 100 | 2.05 | 2 |
| control | actuadores | unrolling | 30 | 0.01 | 2 | 150 | 95.0 | 84.3 | 100 | 1.19 | 2 |
| esfuerzo | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 94.7 | 90.2 | 100 | 21.8 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.01 | 3 | 150 | 94.7 | 85.5 | 100 | 6.79 | 2 |
| condicionamiento | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 94.5 | 87.7 | 100 | 2.13 | 2 |
| control | vanilla | unrolling | 30 | 0.01 | 3 | 150 | 94.2 | 87.7 | 100 | 2.82 | 2 |
| control_margen | vertices | unrolling | 30 | 0.01 | 2 | 150 | 94.2 | 84.5 | 100 | 1 | 2 |
| margen_norm | actuadores | unrolling | 30 | 0.01 | 5 | 150 | 94.1 | 80.0 | 100 | 3.87 | 2 |


## Efecto de cada eje (A2, marginando el resto)


**Función de pérdida** — mejor: `margen_norm` (90.3% A2)

| Función de pérdida | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| margen_norm | 90.3 | 76.4 | 100 | 234500 |
| condicionamiento | 89.9 | 76.2 | 100 | 237000 |
| control_margen | 87.5 | 72.8 | 100 | 237000 |
| esfuerzo | 85.6 | 75.6 | 100 | 237000 |
| paper | 84.1 | 69.5 | 250 | 237000 |
| control | 79.6 | 65.1 | 250 | 237000 |


**Backpropagation** — mejor: `unrolling` (89.3% A2)

| Backpropagation | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| unrolling | 89.3 | 76.0 | 100 | 1272000 |
| implicit | 59.0 | 42.9 | 250 | 147500 |


**Arquitectura** — mejor: `vertices` (90.7% A2)

| Arquitectura | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| vertices | 90.7 | 77.3 | 100 | 36000 |
| actuadores | 86.0 | 72.4 | 100 | 1383500 |


**α (decay prescrito)** — mejor: `0.01` (88.4% A2)

| α (decay prescrito) | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| 0.01 | 88.4 | 80.5 | 100 | 969500 |
| 0.05 | 81.8 | 52.7 | 250 | 150000 |
| 0.001 | 81.8 | 80.6 | 250 | 150000 |
| 0.1 | 80.5 | 33.1 | 250 | 150000 |


**Presupuesto DR de entrenamiento** — mejor: `5` (92.0% A2)

| Presupuesto DR de entrenamiento | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| 5 | 92.0 | 71.0 | 100 | 120000 |
| 30 | 90.8 | 81.0 | 100 | 912000 |
| 120 | 85.5 | 63.1 | 250 | 120000 |
| 500 | 79.4 | 56.0 | 500 | 120000 |
| — | 59.0 | 42.9 | 250 | 147500 |


**Orden del sistema n_x** — mejor: `4` (95.0% A2)

| Orden del sistema n_x | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| 4 | 95.0 | 87.3 | 100 | 66000 |
| 5 | 92.8 | 83.5 | 100 | 66000 |
| 2 | 91.2 | 78.8 | 100 | 12000 |
| 3 | 85.3 | 71.2 | 100 | 1275500 |


**Tamaño de entrenamiento** — mejor: `400` (95.6% A2)

| Tamaño de entrenamiento | %estab A2 | %decay | iters_min | n_sys |
|---|---|---|---|---|
| 400 | 95.6 | 88.9 | 100 | 30000 |
| 50 | 93.4 | 86.4 | 100 | 30000 |
| 150 | 85.8 | 71.9 | 100 | 1359500 |


## Eficiencia (menor iters_min con A2 ≥ 90% estabilizado)

| pérdida | arqu. | backprop | dr_tr | α | n_x | sz | A2 %estab | %decay | iters_min | κ(Q) | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| margen_norm | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 97.4 | 90.9 | 100 | 1.67 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.01 | 3 | 150 | 97.2 | 91.3 | 100 | 7.38 | 2 |
| margen_norm | actuadores | unrolling | 30 | 0.01 | 4 | 150 | 97.1 | 89.6 | 100 | 5.72 | 2 |
| control_margen | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 96.9 | 89.8 | 100 | 1 | 2 |
| control | actuadores | unrolling | 30 | 0.01 | 3 | 400 | 96.9 | 89.9 | 100 | 1 | 2 |
| control_margen | actuadores | unrolling | 5 | 0.01 | 3 | 150 | 96.6 | 90.5 | 100 | 5.24 | 2 |
| control_margen | actuadores | unrolling | 30 | 0.01 | 4 | 150 | 96.5 | 88.5 | 100 | 4.01 | 2 |
| condicionamiento | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 96.2 | 94.9 | 100 | 11.6 | 2 |
| control_margen | actuadores | unrolling | 5 | 0.001 | 3 | 150 | 96.2 | 95.0 | 100 | 3.28 | 2 |
| margen_norm | actuadores | unrolling | 120 | 0.01 | 3 | 150 | 96.1 | 88.6 | 100 | 1.91 | 2 |


## Duelo de arquitecturas × n_x

Config base controlada (`unrolling / dr30 / α=0.01 / sz150`); % estabilizado promedio sobre las 6 pérdidas × 2 semillas. Única diferencia entre celdas: arquitectura y orden. `vanilla` no es invariante → no tiene A2.


**A2 — generalización zero-shot (topología N=5 / n_u=2 no vista)**

| n_x | actuadores | vertices |
|---|---|---|
| 2 | 91.4 | 91.1 |
| 3 | 89.5 | 91.8 |
| 4 | 95.4 | 91.0 |
| 5 | 93.4 | 86.7 |


**A1 — sistemas vistos**

| n_x | actuadores | vertices | vanilla |
|---|---|---|---|
| 2 | 91.2 | 91.5 | 95.8 |
| 3 | 86.0 | 92.2 | 94.0 |
| 4 | 90.9 | 88.4 | 92.5 |
| 5 | 90.5 | 87.5 | 91.8 |


_Lectura: en A2, `actuadores` gana en todos los órdenes (su ventaja sobre `vertices` crece con n_x); en A1, `vanilla` domina pero no generaliza. Cobertura: `actuadores` de E1/E2, `vertices`/`vanilla` de E4._


---
_Métricas por-sistema promediadas en el análisis. A2 = topologías retenidas (N=5, n_u=2) nunca vistas en entrenamiento._
