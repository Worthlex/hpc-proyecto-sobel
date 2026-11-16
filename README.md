# Proyecto final HPC — Filtro Sobel

Proyecto final de Computación de Alto Desempeño (UTP, 2026-2).

## Integrantes

- Samuel Santa Durán
- Miguel Angel Gomez Fernandez

## Objetivo

Partir de una implementación serial correcta del filtro Sobel, construir una versión paralela con memoria compartida usando Numba y `prange`, verificar su correctitud y medir el speedup de forma reproducible.

## Roles y rotación

### Etapa inicial
- Miguel: implementador. Desarrolló la línea base serial y el modelo de referencia de la propuesta.
- Samuel: medidor. Verificó la base, preparó pruebas y analizó las primeras mediciones.

### Etapa final — rotación
- Samuel: implementador. Integró la versión paralela, las pruebas de correctitud y el benchmark reproducible.
- Miguel: medidor. Ejecutó la validación y las mediciones finales en su máquina y revisó los resultados que se usarán en el informe.

## Estructura

- `src/referencia.py`: modelo de referencia NumPy.
- `src/v0_serial.py`: línea base serial en Python.
- `src/v1_numba_serial.py`: control serial compilado con Numba.
- `src/v1_prange.py`: versión paralela con `prange`.
- `test_correctitud.py`: verifica todas las versiones contra la referencia con `rtol=atol=1e-10`.
- `bench.py`: benchmark final con repeticiones, medianas, speedups y metadatos.
- `bench_diagnostico.py`: diagnóstico auxiliar de JIT, tamaño y escalado.
- `resultados/`: resultados crudos generados por el benchmark.
- `reproducir.sh`: recrea el entorno, valida correctitud y regenera los resultados.
- `informe.pdf`: informe final de rendimiento.

## Reproducción

Desde la raíz del repositorio:

```bash
bash reproducir.sh
```

El script debe terminar con el test de correctitud en `PASS` y generar:

```text
resultados/benchmark.json
```

Las cifras del informe final deben salir de ese JSON y corresponder a una sola máquina de medición final.
