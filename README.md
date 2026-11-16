# Proyecto final HPC — Filtro Sobel

Proyecto final de Computación de Alto Desempeño (UTP, 2026-2).

Integrantes:
- Samuel Santa Durán
- Miguel Angel Gomez Fernandez

Objetivo: partir de una implementación serial del filtro Sobel, construir una versión paralela correcta y reproducible, medir su rendimiento y documentar el speedup obtenido.

## Estructura prevista

- `src/referencia.py`: referencia de correctitud.
- `src/v0_serial.py`: línea base serial.
- `src/v1_prange.py`: versión paralela (se añadirá durante el desarrollo).
- `test_correctitud.py`: verificación automática contra la referencia.
- `bench.py`: medición reproducible con repeticiones y metadatos.
- `resultados/`: resultados crudos.
- `reproducir.sh`: regeneración de pruebas y mediciones.
- `informe.pdf`: informe final de rendimiento.
