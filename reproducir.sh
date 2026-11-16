#!/usr/bin/env bash
set -euo pipefail

uv sync
uv run python test_correctitud.py

rm -f resultados/benchmark.json
uv run python bench.py

test -f resultados/benchmark.json
echo "OK: resultados/benchmark.json regenerado"
