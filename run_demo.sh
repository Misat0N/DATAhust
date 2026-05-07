#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "${PROJECT_ROOT}/.venv" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

cd "${PROJECT_ROOT}"
python3 -m streamlit run src/demo/app.py
