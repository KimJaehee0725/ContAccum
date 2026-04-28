#!/usr/bin/env bash
set -euo pipefail

##### env-overridable settings #####
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_CONFIG_DIR="${SCRIPT_DIR}/config"
RUNTIME_CONFIG_FILE="${DOCKER_CONFIG_DIR}/runtime.env"
TOKENS_FILE="${DOCKER_CONFIG_DIR}/.tokens"

if [[ ! -f "${RUNTIME_CONFIG_FILE}" ]]; then
  echo "Error: missing runtime config file: ${RUNTIME_CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! -f "${TOKENS_FILE}" ]]; then
  echo "Error: missing token config file: ${TOKENS_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${RUNTIME_CONFIG_FILE}"
set +a

IMAGE_NAME="${IMAGE_NAME:?IMAGE_NAME must be set in ${RUNTIME_CONFIG_FILE}}"
CONTAINER_NAME="${CONTAINER_NAME:?CONTAINER_NAME must be set in ${RUNTIME_CONFIG_FILE}}"
AUTO_RECREATE="${AUTO_RECREATE:-0}"

# Whitespace- or comma-separated lists.
# Example:
#   PORTS="9204:9204 7860:7860"
#   VOLUMES="/home/jaeheekim/codes:/workspace /media/data:/data"
PORTS="${PORTS:?PORTS must be set in ${RUNTIME_CONFIG_FILE}}"
VOLUMES="${VOLUMES:?VOLUMES must be set in ${RUNTIME_CONFIG_FILE}}"
EXTRA_VOLUMES="${EXTRA_VOLUMES:-}"
WORKSPACE_DIR="${WORKSPACE_DIR:?WORKSPACE_DIR must be set in ${RUNTIME_CONFIG_FILE}}"
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

MOUNT_ARGS+=(-v "${TOKENS_FILE}:${CONTAINER_HOME}/.config/dev-tokens/.tokens:ro")

# If the container already exists, reuse it.
if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Reusing existing container: ${CONTAINER_NAME}" >&2
  echo "Note: changed VOLUMES/PORTS/auth mounts only apply after removing and recreating the container." >&2

  if [[ "${AUTO_RECREATE}" == "1" ]]; then
    docker rm -f "${CONTAINER_NAME}" >/dev/null
  else
    if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
      echo "Error: container exists in docker ps output but cannot be inspected: ${CONTAINER_NAME}" >&2
      exit 1
    fi
    if ! docker start "${CONTAINER_NAME}" >/dev/null; then
      echo "Error: failed to start existing container: ${CONTAINER_NAME}" >&2
      echo "Remove it and rerun: docker rm -f ${CONTAINER_NAME}" >&2
      exit 1
    fi
    if ! docker exec -w / "${CONTAINER_NAME}" test -d "${WORKSPACE_DIR}"; then
      old_workdir="$(docker inspect -f '{{.Config.WorkingDir}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
      echo "Error: WORKSPACE_DIR does not exist inside existing container: ${WORKSPACE_DIR}" >&2
      if [[ -n "${old_workdir}" ]]; then
        echo "Existing container image/workdir: ${old_workdir}" >&2
      fi
      echo "This usually means the container was created with old volume/workdir settings." >&2
      echo "Remove and recreate it: docker rm -f ${CONTAINER_NAME} && bash ${BASH_SOURCE[0]}" >&2
      echo "Or run once with AUTO_RECREATE=1: AUTO_RECREATE=1 bash ${BASH_SOURCE[0]}" >&2
      exit 1
    fi
  fi
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
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
