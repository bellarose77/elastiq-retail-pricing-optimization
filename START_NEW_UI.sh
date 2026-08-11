#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
printf 'Starting ELASTIQ DEEP ANALYSIS RUNNER - BUILD 1.2.0\n'
exec ./start.sh "$@"
