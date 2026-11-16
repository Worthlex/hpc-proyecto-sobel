"""Benchmark diagnostico: separa JIT de paralelismo y compara tamaños."""

from __future__ import annotations

import statistics
import time

import numba

from src.v0_serial import crear_imagen_prueba, sobel_serial
from src.v1_numba_serial import sobel_numba_serial
from src.v1_prange import sobel_prange


TAMANOS = (1000, 2000, 4000)
REPETICIONES = 7
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


def medir_tamano(n):
    imagen = crear_imagen_prueba(n)

    # Calentamiento JIT fuera de las mediciones.
    sobel_numba_serial(imagen)
    numba.set_num_threads(1)
    sobel_prange(imagen)

    print(f"\n=== Imagen {n}x{n} ===")
    print(f"Repeticiones por punto: {REPETICIONES}")

    # Python puro solo se mide en 1000x1000 para no alargar innecesariamente
    # el diagnostico con tamaños grandes.
    t_python = None
    if n == 1000:
        print("Midiendo Python serial...")
        t_python = mediana(medir(sobel_serial, imagen))

    print("Midiendo Numba serial...")
    t_numba = mediana(medir(sobel_numba_serial, imagen))

    resultados_prange = {}

    for hilos in HILOS:
        numba.set_num_threads(hilos)

        # Estabilizacion tras cambiar el numero de hilos.
        sobel_prange(imagen)

        print(f"Midiendo prange con {hilos} hilo(s)...")
        resultados_prange[hilos] = mediana(
            medir(sobel_prange, imagen)
        )

    print("\nResultados:")
    if t_python is not None:
        print(f"Python serial : {t_python:.6f} s")
        print(
            f"Ganancia JIT  : {t_python / t_numba:.2f}x "
            "(Python / Numba serial)"
        )

    print(f"Numba serial  : {t_numba:.6f} s")

    t_prange_1 = resultados_prange[1]

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
        f"Mejor prange: {mejor_hilos} hilo(s), "
        f"{mejor_tiempo:.6f} s"
    )
    print(
        f"Speedup paralelo real aproximado: "
        f"{t_numba / mejor_tiempo:.2f}x"
    )

    return t_numba, resultados_prange


def main():
    print(
        "Diagnostico Sobel: Numba serial vs prange "
        "con 1, 2, 4 y 8 hilos"
    )

    for n in TAMANOS:
        medir_tamano(n)


if __name__ == "__main__":
    main()
