"""
backprop_unrolling.py
=====================
Estrategia de backpropagation por UNROLLING (desenrollado).

Se itera Douglas-Rachford `dr_iters` veces y el gradiente fluye a traves de
TODAS las iteraciones: PyTorch guarda el grafo completo del bucle. Por eso el
gradiente depende del nº de iteraciones y la memoria crece O(dr_iters).

Es una estrategia intercambiable: se combina con cualquier arquitectura
(vanilla / vertices / actuadores). La iteracion de DR en si es la MISMA que en la
version implicita (esta en core._dr_iterate); lo unico que cambia aqui es que NO
se desactiva el grafo.
"""


class UnrollingSolver:
    name = "unrolling"

    def project(self, model, y_hat, A_poly, B_poly):
        L, c, M_inv = model._dr_precompute(A_poly, B_poly)
        # CON grafo: el gradiente se retropropaga por las dr_iters iteraciones
        y_k, x_k = model._dr_iterate(y_hat, L, c, M_inv, model.dr_iters)
        y_star = model._dr_final_proj(y_hat, y_k, x_k, L, c, M_inv)
        return model._y_to_matrices(y_star)
