import numpy as np


def sobel_serial(imagen):
    alto, ancho = imagen.shape
    salida = np.zeros((alto, ancho), dtype=np.float64)

    for i in range(1, alto - 1):
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


def crear_imagen_prueba(n=1000):
    rng = np.random.default_rng(42)
    return rng.random((n, n), dtype=np.float64)


if __name__ == "__main__":
    imagen = crear_imagen_prueba()
    resultado = sobel_serial(imagen)

    print("Forma:", resultado.shape)
    print("Suma:", float(resultado.sum()))
