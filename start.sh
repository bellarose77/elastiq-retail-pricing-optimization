#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$APP_ROOT/app/frontend"
RUNTIME_DIR="$APP_ROOT/.runtime"
export npm_config_cache="$RUNTIME_DIR/npm-cache"

printf '\n============================================================\n'
printf '  ELASTIQ Retail Pricing Decision Studio\n'
printf '  DEEP ANALYSIS RUNNER - BUILD 1.2.0\n'
printf '============================================================\n\n'

if ! command -v node >/dev/null 2>&1; then
  printf '[ERROR] Node.js is not installed.\n'
  printf 'Install Node.js 18 or newer from https://nodejs.org and run ./start.sh again.\n'
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  printf '[ERROR] npm is not available. Reinstall Node.js and run ./start.sh again.\n'
  exit 1
fi

mkdir -p "$RUNTIME_DIR"

if [[ "${1:-}" == "--refresh-data" ]]; then
  printf '[1/3] Preparing the Python analytics environment...\n'
  if ! command -v python3 >/dev/null 2>&1; then
    printf '[ERROR] Python 3 is required for --refresh-data.\n'
    printf 'Install Python 3.10 or newer, or run ./start.sh without --refresh-data.\n'
    exit 1
  fi
  if [[ ! -x "$RUNTIME_DIR/venv/bin/python" ]]; then
    python3 -m venv "$RUNTIME_DIR/venv"
  fi
  "$RUNTIME_DIR/venv/bin/python" -m pip install --disable-pip-version-check -r "$APP_ROOT/requirements.txt"
  "$RUNTIME_DIR/venv/bin/python" "$APP_ROOT/scripts/run_pipeline.py"
fi

printf '[2/3] Preparing the web application...\n'
cd "$WEB_DIR"
if [[ ! -f node_modules/.package-lock.json ]]; then
  npm --cache "$npm_config_cache" ci --no-audit --no-fund
fi

printf '[3/3] Starting ELASTIQ at http://127.0.0.1:5173\n'
printf 'Keep this terminal open while using the application. Press Ctrl+C to stop it.\n\n'

if ! node -e 'const net=require("net");const s=net.createServer();s.once("error",()=>process.exit(1));s.once("listening",()=>s.close(()=>process.exit(0)));s.listen(5173,"127.0.0.1")'; then
  printf '[ERROR] Port 5173 is already in use.\n'
  printf 'Close the earlier ELASTIQ terminal, then run ./start.sh again.\n'
  exit 1
fi

if [[ "${1:-}" != "--no-browser" ]]; then
  (
    sleep 2
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open 'http://127.0.0.1:5173/?build=DEEP-ANALYSIS-1.2.0' >/dev/null 2>&1 || true
    elif command -v open >/dev/null 2>&1; then
      open 'http://127.0.0.1:5173/?build=DEEP-ANALYSIS-1.2.0' >/dev/null 2>&1 || true
    fi
  ) &
fi

exec npm run dev -- --host 127.0.0.1 --strictPort --force
