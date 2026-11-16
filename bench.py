"""Benchmark reproducible del proyecto Sobel."""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from pathlib import Path

import numba
import numpy as np

from src.v0_serial import crear_imagen_prueba, sobel_serial
from src.v1_prange import sobel_prange


N = 1000
REPETICIONES = 5
RUTA_RESULTADOS = Path("resultados/benchmark.json")


def medir(funcion, imagen, repeticiones=REPETICIONES):
    tiempos = []

    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion(imagen)
        fin = time.perf_counter()
        tiempos.append(fin - inicio)

    return tiempos


def resumen(tiempos):
    return {
        "tiempos_s": tiempos,
        "mediana_s": statistics.median(tiempos),
        "min_s": min(tiempos),
        "max_s": max(tiempos),
    }


def metadatos():
    return {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "procesador": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "numba_threads": numba.get_num_threads(),
    }


def main():
    imagen = crear_imagen_prueba(N)

    # Calentamiento para excluir el costo de compilacion JIT de Numba.
    sobel_prange(imagen)

    print(f"Imagen: {N}x{N}")
    print(f"Repeticiones: {REPETICIONES}")
    print(f"Hilos Numba: {numba.get_num_threads()}")

    print("\nMidiendo v0_serial...")
    tiempos_serial = medir(sobel_serial, imagen)

    print("Midiendo v1_prange...")
    tiempos_prange = medir(sobel_prange, imagen)

    serial = resumen(tiempos_serial)
    prange = resumen(tiempos_prange)

    speedup = serial["mediana_s"] / prange["mediana_s"]

    resultado = {
        "configuracion": {
            "tamano": [N, N],
            "repeticiones": REPETICIONES,
            "seed": 42,
        },
        "metadatos": metadatos(),
        "v0_serial": serial,
        "v1_prange": prange,
        "speedup_mediana": speedup,
    }

    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(
        json.dumps(resultado, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nSerial mediana: {serial['mediana_s']:.6f} s"
    )
    print(
        f"Prange mediana: {prange['mediana_s']:.6f} s"
    )
    print(f"Speedup: {speedup:.2f}x")
    print(f"Resultados: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
