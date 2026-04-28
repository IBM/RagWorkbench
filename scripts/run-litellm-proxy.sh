#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE_DIR="${PROJECT_ROOT}/src/ragworkbench/litellm_proxy"
POSTGRES_DATA_DIR="${COMPOSE_DIR}/postgres_data"

mkdir -p "${POSTGRES_DATA_DIR}"

docker compose \
  --env-file "${COMPOSE_DIR}/.env" \
  -f "${COMPOSE_DIR}/docker-compose.yml" \
  up -d
