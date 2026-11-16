"""Benchmark final reproducible del proyecto Sobel."""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np

from src.v0_serial import crear_imagen_prueba, sobel_serial
from src.v1_numba_serial import sobel_numba_serial
from src.v1_prange import sobel_prange


N = 4000
REPETICIONES = 5
HILOS = (1, 2, 4, 8)
RUTA_RESULTADOS = Path("resultados/benchmark.json")


def medir(funcion, imagen, repeticiones=REPETICIONES):
    tiempos = []

    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion(imagen)
        tiempos.append(time.perf_counter() - inicio)

    return tiempos


def resumen(tiempos):
    return {
        "tiempos_s": tiempos,
        "mediana_s": statistics.median(tiempos),
        "min_s": min(tiempos),
        "max_s": max(tiempos),
    }


def cpu_model():
    try:
        for linea in Path("/proc/cpuinfo").read_text(
            encoding="utf-8"
        ).splitlines():
            if linea.lower().startswith("model name"):
                return linea.split(":", 1)[1].strip()
    except OSError:
        pass

    return platform.processor() or "desconocido"


def memoria_total_gib():
    try:
        for linea in Path("/proc/meminfo").read_text(
            encoding="utf-8"
        ).splitlines():
            if linea.startswith("MemTotal:"):
                kib = int(linea.split()[1])
                return kib / 1024 / 1024
    except (OSError, ValueError):
        pass

    return None


def metadatos():
    return {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "cpu": cpu_model(),
        "cpu_logicos": os.cpu_count(),
        "memoria_total_gib": memoria_total_gib(),
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "numba_max_threads": numba.config.NUMBA_NUM_THREADS,
    }


def main():
    imagen = crear_imagen_prueba(N)

    # Calentamiento JIT fuera de las mediciones.
    sobel_numba_serial(imagen)
    numba.set_num_threads(1)
    sobel_prange(imagen)

    print(f"Imagen: {N}x{N}")
    print(f"Repeticiones por punto: {REPETICIONES}")

    print("\nMidiendo v0_serial (Python)...")
    serial_python = resumen(
        medir(sobel_serial, imagen)
    )

    print("Midiendo v1_numba_serial...")
    serial_numba = resumen(
        medir(sobel_numba_serial, imagen)
    )

    prange = {}

    for hilos in HILOS:
        numba.set_num_threads(hilos)

        # Estabilizacion después de cambiar el número de hilos.
        sobel_prange(imagen)

        print(f"Midiendo prange con {hilos} hilo(s)...")
        datos = resumen(
            medir(sobel_prange, imagen)
        )
        prange[str(hilos)] = datos

    t_python = serial_python["mediana_s"]
    t_numba = serial_numba["mediana_s"]
    t_prange_1 = prange["1"]["mediana_s"]

    for hilos in HILOS:
        datos = prange[str(hilos)]
        t = datos["mediana_s"]

        datos["speedup_absoluto_vs_python"] = t_python / t
        datos["speedup_relativo_vs_prange1"] = t_prange_1 / t
        datos["speedup_vs_numba_serial"] = t_numba / t

    mejor_hilos = min(
        HILOS,
        key=lambda h: prange[str(h)]["mediana_s"],
    )

    resultado = {
        "configuracion": {
            "tamano": [N, N],
            "repeticiones": REPETICIONES,
            "seed": 42,
            "hilos_probados": list(HILOS),
            "rtol_correctitud": 1e-10,
            "atol_correctitud": 1e-10,
        },
        "maquina": metadatos(),
        "v0_serial_python": serial_python,
        "v1_numba_serial": serial_numba,
        "v1_prange": prange,
        "mejor_prange": {
            "hilos": mejor_hilos,
            "mediana_s": prange[str(mejor_hilos)]["mediana_s"],
            "speedup_absoluto_vs_python":
                prange[str(mejor_hilos)]["speedup_absoluto_vs_python"],
            "speedup_relativo_vs_prange1":
                prange[str(mejor_hilos)]["speedup_relativo_vs_prange1"],
            "speedup_vs_numba_serial":
                prange[str(mejor_hilos)]["speedup_vs_numba_serial"],
        },
    }

    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.write_text(
        json.dumps(resultado, indent=2),
        encoding="utf-8",
    )

    print("\n--- RESUMEN ---")
    print(
        f"Python serial : {serial_python['mediana_s']:.6f} s"
    )
    print(
        f"Numba serial  : {serial_numba['mediana_s']:.6f} s"
    )

    for hilos in HILOS:
        datos = prange[str(hilos)]
        print(
            f"Prange {hilos:2d}: {datos['mediana_s']:.6f} s  "
            f"abs={datos['speedup_absoluto_vs_python']:.2f}x  "
            f"rel={datos['speedup_relativo_vs_prange1']:.2f}x  "
            f"vs_numba={datos['speedup_vs_numba_serial']:.2f}x"
        )

    mejor = resultado["mejor_prange"]
    print(
        f"\nMejor configuración: {mejor['hilos']} hilos"
    )
    print(
        f"Speedup absoluto vs Python: "
        f"{mejor['speedup_absoluto_vs_python']:.2f}x"
    )
    print(
        f"Speedup relativo vs prange(1): "
        f"{mejor['speedup_relativo_vs_prange1']:.2f}x"
    )
    print(
        f"Speedup vs Numba serial: "
        f"{mejor['speedup_vs_numba_serial']:.2f}x"
    )
    print(f"Resultados: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
