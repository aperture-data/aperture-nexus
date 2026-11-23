#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SDK_WHEEL_DIR=$(mktemp -d)
export SDK_WHEEL_DIR
trap 'rm -rf "$SDK_WHEEL_DIR"' EXIT
oras pull registry-1.docker.io/aperturedata/aperturedb-python:latest --output "$SDK_WHEEL_DIR"
docker compose --profile demo run --rm --build nexus-demo "$@"
