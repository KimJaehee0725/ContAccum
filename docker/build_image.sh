#!/usr/bin/env bash
set -euo pipefail

##### env-overridable settings #####
IMAGE_NAME="${IMAGE_NAME:-jaehee-base:0404}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-/home/jaeheekim/codes/ContAccum/docker/Dockerfile}"
BUILD_CONTEXT_DIR="${BUILD_CONTEXT_DIR:-/home/jaeheekim/codes}"
NO_CACHE="${NO_CACHE:-0}"
####################################

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER="$(whoami)"

GIT_NAME="$(git config --global user.name 2>/dev/null || echo "Jaehee Kim")"
GIT_EMAIL="$(git config --global user.email 2>/dev/null || echo "jaehee_kim@snu.ac.kr")"

BUILD_FLAGS=(--progress=plain)
if [[ "${NO_CACHE}" == "1" ]]; then
  BUILD_FLAGS+=(--no-cache)
fi

docker build \
  "${BUILD_FLAGS[@]}" \
  --build-arg UID="${HOST_UID}" \
  --build-arg GID="${HOST_GID}" \
  --build-arg USERNAME="${HOST_USER}" \
  --build-arg GIT_NAME="${GIT_NAME}" \
  --build-arg GIT_EMAIL="${GIT_EMAIL}" \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE_PATH}" \
  "${BUILD_CONTEXT_DIR}"
