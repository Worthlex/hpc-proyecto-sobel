import numpy as np
from numba import njit, prange


@njit(parallel=True)
def sobel_prange(imagen):
    alto, ancho = imagen.shape
    salida = np.zeros((alto, ancho), dtype=np.float64)

    for i in prange(1, alto - 1):
        for j in range(1, ancho - 1):
            gx = (
                -imagen[i - 1, j - 1]
                + imagen[i - 1, j + 1]
                - 2 * imagen[i, j - 1]
                + 2 * imagen[i, j + 1]
                - imagen[i + 1, j - 1]
                + imagen[i + 1, j + 1]
            )

            gy = (
                imagen[i - 1, j - 1]
                + 2 * imagen[i - 1, j]
                + imagen[i - 1, j + 1]
                - imagen[i + 1, j - 1]
                - 2 * imagen[i + 1, j]
                - imagen[i + 1, j + 1]
            )

            salida[i, j] = np.sqrt(gx * gx + gy * gy)

    return salida
