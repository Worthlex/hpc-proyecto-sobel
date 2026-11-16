import numpy as np

from .v0_serial import crear_imagen_prueba, sobel_serial


def sobel_numpy(imagen):
    gx = (
        -imagen[:-2, :-2]
        + imagen[:-2, 2:]
        - 2 * imagen[1:-1, :-2]
        + 2 * imagen[1:-1, 2:]
        - imagen[2:, :-2]
        + imagen[2:, 2:]
    )

    gy = (
        imagen[:-2, :-2]
        + 2 * imagen[:-2, 1:-1]
        + imagen[:-2, 2:]
        - imagen[2:, :-2]
        - 2 * imagen[2:, 1:-1]
        - imagen[2:, 2:]
    )

    salida = np.zeros_like(imagen, dtype=np.float64)
    salida[1:-1, 1:-1] = np.sqrt(gx * gx + gy * gy)

    return salida


if __name__ == "__main__":
    imagen = crear_imagen_prueba(1000)

    resultado_serial = sobel_serial(imagen)
    resultado_numpy = sobel_numpy(imagen)

    print(
        "Coinciden:",
        np.allclose(
            resultado_serial,
            resultado_numpy,
            rtol=1e-10,
            atol=1e-10,
        ),
    )

    print("Suma serial:", float(resultado_serial.sum()))
    print("Suma NumPy:", float(resultado_numpy.sum()))
