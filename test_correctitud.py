"""Verificación de correctitud del proyecto Sobel."""

import numpy as np

from src.referencia import sobel_numpy
from src.v0_serial import crear_imagen_prueba, sobel_serial


RTOL = 1e-10
ATOL = 1e-10
TAMANOS = (64, 257, 1000)


def verificar(n):
    imagen = crear_imagen_prueba(n)

    esperado = sobel_numpy(imagen)
    obtenido = sobel_serial(imagen)

    error_max = float(np.max(np.abs(obtenido - esperado)))
    correcto = np.allclose(
        obtenido,
        esperado,
        rtol=RTOL,
        atol=ATOL,
    )

    print(
        f"n={n:4d}  correcto={correcto}  "
        f"error_max={error_max:.3e}"
    )

    if not correcto:
        raise AssertionError(
            f"v0_serial no coincide con la referencia para n={n}"
        )


def main():
    print(
        f"Tolerancia: rtol={RTOL:g}, atol={ATOL:g}"
    )

    for n in TAMANOS:
        verificar(n)

    print("PASS: v0_serial coincide con la referencia")


if __name__ == "__main__":
    main()
