"""Benchmark diagnostico: separa JIT de paralelismo."""

from __future__ import annotations

import statistics
import time

import numba

from src.v0_serial import crear_imagen_prueba, sobel_serial
from src.v1_numba_serial import sobel_numba_serial
from src.v1_prange import sobel_prange


N = 1000
REPETICIONES = 9
HILOS = (1, 2, 4, 8)


def medir(funcion, imagen, repeticiones=REPETICIONES):
    tiempos = []

    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion(imagen)
        tiempos.append(time.perf_counter() - inicio)

    return tiempos


def mediana(tiempos):
    return statistics.median(tiempos)


def main():
    imagen = crear_imagen_prueba(N)

    # Calentamiento JIT.
    sobel_numba_serial(imagen)
    numba.set_num_threads(1)
    sobel_prange(imagen)

    print(f"Imagen: {N}x{N}")
    print(f"Repeticiones por punto: {REPETICIONES}")

    print("\nMidiendo Python serial...")
    t_python = mediana(medir(sobel_serial, imagen))

    print("Midiendo Numba serial...")
    t_numba = mediana(medir(sobel_numba_serial, imagen))

    resultados_prange = {}

    for hilos in HILOS:
        numba.set_num_threads(hilos)

        # Una llamada de estabilizacion despues de cambiar hilos.
        sobel_prange(imagen)

        print(f"Midiendo prange con {hilos} hilo(s)...")
        t = mediana(medir(sobel_prange, imagen))
        resultados_prange[hilos] = t

    print("\n--- RESULTADOS ---")
    print(f"Python serial : {t_python:.6f} s")
    print(f"Numba serial  : {t_numba:.6f} s")
    print(
        f"Ganancia JIT  : {t_python / t_numba:.2f}x "
        "(Python / Numba serial)"
    )

    t_prange_1 = resultados_prange[1]

    print("\nPrange:")
    for hilos, tiempo in resultados_prange.items():
        speedup_vs_1 = t_prange_1 / tiempo
        speedup_vs_numba = t_numba / tiempo

        print(
            f"{hilos:2d} hilo(s): {tiempo:.6f} s  "
            f"speedup_vs_prange1={speedup_vs_1:.2f}x  "
            f"speedup_vs_numba_serial={speedup_vs_numba:.2f}x"
        )

    mejor_hilos = min(resultados_prange, key=resultados_prange.get)
    mejor_tiempo = resultados_prange[mejor_hilos]

    print(
        f"\nMejor prange: {mejor_hilos} hilo(s), "
        f"{mejor_tiempo:.6f} s"
    )
    print(
        f"Speedup paralelo real aproximado: "
        f"{t_numba / mejor_tiempo:.2f}x"
    )
    print(
        f"Speedup total vs Python: "
        f"{t_python / mejor_tiempo:.2f}x"
    )


if __name__ == "__main__":
    main()
