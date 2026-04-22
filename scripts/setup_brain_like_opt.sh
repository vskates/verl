#!/usr/bin/env bash
set -euxo pipefail

IMAGE="${IMAGE:-whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3}"
CONTAINER_NAME="${CONTAINER_NAME:-verl}"
HOME_DIR="${HOME}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

mkdir -p "${HOME_DIR}/.cache" \
         "${HOME_DIR}/data" \
         "${HOME_DIR}/runroot/runtime_src" \
         "${HOME_DIR}/runroot/data" \
         "${HOME_DIR}/runroot/checkpoints" \
         "${HOME_DIR}/runroot/verl" \
         "${HOME_DIR}/runroot/hf_cache" \
         "${HOME_DIR}/runroot/outputs"

if [ -d "${HOME_DIR}/verl" ]; then
  rsync -a --delete --exclude '.git' "${HOME_DIR}/verl/" "${HOME_DIR}/runroot/runtime_src/"
fi

"${DOCKER_BIN}" pull "${IMAGE}"
"${DOCKER_BIN}" rm -f "${CONTAINER_NAME}" || true

"${DOCKER_BIN}" run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${HOME_DIR}/verl:/workspace/verl" \
  -v "${HOME_DIR}/runroot:/workspace/runroot" \
  -v "${HOME_DIR}/.cache:/root/.cache" \
  -v "${HOME_DIR}/data:/root/data" \
  "${IMAGE}" \
  -f /dev/null

"${DOCKER_BIN}" exec "${CONTAINER_NAME}" bash -lc '
set -euxo pipefail
mkdir -p /workspace/runroot/runtime_src \
         /workspace/runroot/data \
         /workspace/runroot/checkpoints \
         /workspace/runroot/verl \
         /workspace/runroot/hf_cache \
         /workspace/runroot/outputs
python3 --version
pip show verl | sed -n "1,20p"
'

echo "Container ${CONTAINER_NAME} is ready."
