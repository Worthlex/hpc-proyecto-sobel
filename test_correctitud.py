"""Verificación de correctitud del proyecto Sobel."""

import numpy as np

from src.referencia import sobel_numpy
from src.v0_serial import crear_imagen_prueba, sobel_serial
from src.v1_numba_serial import sobel_numba_serial
from src.v1_prange import sobel_prange


RTOL = 1e-10
ATOL = 1e-10
TAMANOS = (64, 257, 1000)


def verificar(n):
    imagen = crear_imagen_prueba(n)

    esperado = sobel_numpy(imagen)
    obtenido_serial = sobel_serial(imagen)
    obtenido_numba = sobel_numba_serial(imagen)
    obtenido_prange = sobel_prange(imagen)

    error_serial = float(np.max(np.abs(obtenido_serial - esperado)))
    error_numba = float(np.max(np.abs(obtenido_numba - esperado)))
    error_prange = float(np.max(np.abs(obtenido_prange - esperado)))

    correcto_serial = np.allclose(
        obtenido_serial,
        esperado,
        rtol=RTOL,
        atol=ATOL,
    )

    correcto_numba = np.allclose(
        obtenido_numba,
        esperado,
        rtol=RTOL,
        atol=ATOL,
    )

    correcto_prange = np.allclose(
        obtenido_prange,
        esperado,
        rtol=RTOL,
        atol=ATOL,
    )

    print(
        f"n={n:4d}  "
        f"serial={correcto_serial}  "
        f"numba={correcto_numba}  "
        f"prange={correcto_prange}  "
        f"error_serial={error_serial:.3e}  "
        f"error_numba={error_numba:.3e}  "
        f"error_prange={error_prange:.3e}"
    )

    if not correcto_serial:
        raise AssertionError(
            f"v0_serial no coincide con la referencia para n={n}"
        )

    if not correcto_numba:
        raise AssertionError(
            f"v1_numba_serial no coincide con la referencia para n={n}"
        )

    if not correcto_prange:
        raise AssertionError(
            f"v1_prange no coincide con la referencia para n={n}"
        )


def main():
    print(
        f"Tolerancia: rtol={RTOL:g}, atol={ATOL:g}"
    )

    for n in TAMANOS:
        verificar(n)

    print(
        "PASS: v0_serial, v1_numba_serial y v1_prange "
        "coinciden con la referencia"
    )


if __name__ == "__main__":
    main()
