#!/usr/bin/env bash
set -euo pipefail

uv sync
uv run python test_correctitud.py
uv run python bench.py
