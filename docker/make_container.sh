#!/usr/bin/env bash
set -euo pipefail

##### env-overridable settings #####
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_CONFIG_DIR="${SCRIPT_DIR}/config"

IMAGE_NAME="${IMAGE_NAME:-jaehee-base:0404}"
CONTAINER_NAME="${CONTAINER_NAME:-jaehee-contaccum-refine}"

HOST_CODE_DIR="${HOST_CODE_DIR:-/home/jaeheekim/codes}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
HOST_DATA_DIR="${HOST_DATA_DIR:-/media/data}"
CONTAINER_DATA_DIR="${CONTAINER_DATA_DIR:-/data}"

# Whitespace- or comma-separated lists.
# Example:
#   PORTS="9204:9204 7860:7860"
#   VOLUMES="/home/jaeheekim/codes:/workspace /media/data:/data"
PORTS="${PORTS:-9204:9204}"
VOLUMES="${VOLUMES:-${HOST_CODE_DIR}:${WORKSPACE_DIR} ${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}}"
EXTRA_VOLUMES="${EXTRA_VOLUMES:-}"

HOST_GH_CONFIG_DIR="${DOCKER_CONFIG_DIR}/gh"
HOST_GITHUB_CONFIG_DIR="${DOCKER_CONFIG_DIR}/github"
HOST_HUGGINGFACE_CONFIG_DIR="${DOCKER_CONFIG_DIR}/huggingface"
####################################

TERM_VALUE="${TERM:-xterm-256color}"
COLORTERM_VALUE="${COLORTERM:-truecolor}"
TERM_PROGRAM_VALUE="${TERM_PROGRAM:-}"

CONTAINER_HOME="/home/${USER:-$(whoami)}"

ENV_ARGS=(
  -e "TERM=${TERM_VALUE}"
  -e "COLORTERM=${COLORTERM_VALUE}"
  -e "TERM_PROGRAM=${TERM_PROGRAM_VALUE}"
)
MOUNT_ARGS=()
PORT_ARGS=()

add_list_args() {
  local flag="$1"
  local raw="${2//,/ }"
  local item

  for item in ${raw}; do
    [[ -n "${item}" ]] && "$3" "${flag}" "${item}"
  done
}

append_mount_arg() {
  local flag="$1"
  local value="$2"
  MOUNT_ARGS+=("${flag}" "${value}")
}

append_port_arg() {
  local flag="$1"
  local value="$2"
  PORT_ARGS+=("${flag}" "${value}")
}

CONTAINER_COMMAND=(zsh -lc "init-dev-auth >/dev/null 2>&1 || true; exec zsh -l")

validate_volume_hosts() {
  local raw="${1//,/ }"
  local volume host_path

  for volume in ${raw}; do
    [[ -z "${volume}" ]] && continue
    host_path="${volume%%:*}"
    if [[ "${host_path}" = /* && ! -e "${host_path}" ]]; then
      echo "Error: volume host path does not exist: ${host_path}" >&2
      exit 1
    fi
  done
}

validate_volume_hosts "${VOLUMES}"
validate_volume_hosts "${EXTRA_VOLUMES}"
add_list_args -v "${VOLUMES}" append_mount_arg
add_list_args -v "${EXTRA_VOLUMES}" append_mount_arg
add_list_args -p "${PORTS}" append_port_arg

if [[ ! -d "${HOST_GH_CONFIG_DIR}" ]]; then
  echo "Error: missing GitHub CLI config directory: ${HOST_GH_CONFIG_DIR}" >&2
  exit 1
fi
if [[ ! -f "${HOST_GITHUB_CONFIG_DIR}/token" ]]; then
  echo "Error: missing GitHub token file: ${HOST_GITHUB_CONFIG_DIR}/token" >&2
  exit 1
fi
if [[ ! -f "${HOST_HUGGINGFACE_CONFIG_DIR}/token" ]]; then
  echo "Error: missing Hugging Face token file: ${HOST_HUGGINGFACE_CONFIG_DIR}/token" >&2
  exit 1
fi

MOUNT_ARGS+=(-v "${HOST_GH_CONFIG_DIR}:${CONTAINER_HOME}/.config/gh")
MOUNT_ARGS+=(-v "${HOST_GITHUB_CONFIG_DIR}:${CONTAINER_HOME}/.config/github:ro")
MOUNT_ARGS+=(-v "${HOST_HUGGINGFACE_CONFIG_DIR}:${CONTAINER_HOME}/.config/huggingface")

# If the container already exists, reuse it.
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Reusing existing container: ${CONTAINER_NAME}" >&2
  echo "Note: changed VOLUMES/PORTS/auth mounts only apply after removing and recreating the container." >&2
  if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    docker start "${CONTAINER_NAME}" >/dev/null
  fi

  docker exec -it \
    "${ENV_ARGS[@]}" \
    -w "${WORKSPACE_DIR}" \
    "${CONTAINER_NAME}" \
    "${CONTAINER_COMMAND[@]}"

  exit 0
fi

DOCKER_ARGS=(
  -it
  --gpus all
  --ipc=host
  --name "${CONTAINER_NAME}"
  --hostname "${CONTAINER_NAME}"
  "${ENV_ARGS[@]}"
  "${MOUNT_ARGS[@]}"
  "${PORT_ARGS[@]}"
  -w "${WORKSPACE_DIR}"
)

docker run \
  "${DOCKER_ARGS[@]}" \
  "${IMAGE_NAME}" \
  "${CONTAINER_COMMAND[@]}"
