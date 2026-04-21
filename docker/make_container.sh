#!/usr/bin/env bash
set -euo pipefail

##### user-editable settings #####
IMAGE_NAME="jaehee-base:0404"
CONTAINER_NAME="jaehee-contaccum-refine"

# Code volume:
#   HOST_CODE_DIR -> /workspace
# Data volume:
#   HOST_DATA_DIR -> /data
HOST_CODE_DIR="/home/jaeheekim/codes/personnal_study"
WORKSPACE_DIR="/workspace"

HOST_DATA_DIR="/data3/jaehee"
CONTAINER_DATA_DIR="/data"

PORTS=(
  "9204:9204"
)
##################################

TERM_VALUE="${TERM:-xterm-256color}"
COLORTERM_VALUE="${COLORTERM:-truecolor}"
TERM_PROGRAM_VALUE="${TERM_PROGRAM:-}"

if [[ ! -d "${HOST_CODE_DIR}" ]]; then
  echo "Error: HOST_CODE_DIR does not exist: ${HOST_CODE_DIR}" >&2
  exit 1
fi

if [[ ! -d "${HOST_DATA_DIR}" ]]; then
  echo "Error: HOST_DATA_DIR does not exist: ${HOST_DATA_DIR}" >&2
  exit 1
fi

# If the container already exists, reuse it.
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    docker start "${CONTAINER_NAME}" >/dev/null
  fi

  docker exec -it \
    -e "TERM=${TERM_VALUE}" \
    -e "COLORTERM=${COLORTERM_VALUE}" \
    -e "TERM_PROGRAM=${TERM_PROGRAM_VALUE}" \
    -w "${WORKSPACE_DIR}" \
    "${CONTAINER_NAME}" \
    zsh -l

  exit 0
fi

DOCKER_ARGS=(
  -it
  --gpus all
  --ipc=host
  --name "${CONTAINER_NAME}"
  --hostname "${CONTAINER_NAME}"
  -e "TERM=${TERM_VALUE}"
  -e "COLORTERM=${COLORTERM_VALUE}"
  -e "TERM_PROGRAM=${TERM_PROGRAM_VALUE}"
  -v "${HOST_CODE_DIR}:${WORKSPACE_DIR}"
  -v "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}"
  -w "${WORKSPACE_DIR}"
)

for port_map in "${PORTS[@]}"; do
  DOCKER_ARGS+=(-p "${port_map}")
done

docker run \
  "${DOCKER_ARGS[@]}" \
  "${IMAGE_NAME}" \
  zsh -l
