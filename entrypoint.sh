#!/usr/bin/env bash
#
# Container entrypoint: start basal-serve on loopback, then hand the process to
# the RunPod handler.
#
# The engine runs in the background because RunPod owns the foreground process
# (the handler must be it: when the handler exits, the job's container goes with
# it). The handler then waits for `GET /health` before serving a batch, so a
# slow model load shows up as `worker_unhealthy` on the first job rather than as
# a crashed container.
#
# Every knob is an environment variable; none of them carries a secret, and none
# of them is a URL to anything outside the container.
set -euo pipefail

model="${BASAL_MODEL:-Remek/basal-omni-1.0-mini}"
mode="${BASAL_MODE:-fast-nocompile}"
media="${BASAL_MEDIA:-off}"
port="${BASAL_PORT:-8001}"
calibration="${BASAL_CALIBRATION:-/calibration/CALIBRATION.custom.json}"

args=(--model "${model}" --host 127.0.0.1 --port "${port}")

if [ "${media}" = "on" ]; then
  # `--media on` is an eager-mode feature: the CUDA-graph engine does not load
  # the image or audio towers (docs/OMNI.md).
  args+=(--mode eager --media on)
else
  args+=(--mode "${mode}")
fi

# A state that does not fit the context is refused (422) instead of silently
# truncated: a cut-off state produces a confident wrong decision, and the
# gateway can split the request where the engine cannot.
args+=(--context-overflow refuse)

if [ -f "${calibration}" ]; then
  # `--strict-calibration` refuses a file fitted on another model or revision
  # instead of quietly applying the wrong temperatures.
  args+=(--calibration "${calibration}" --strict-calibration)
fi

# Operator escape hatch for engine tuning (--max-len, --orders, …). Split on
# whitespace on purpose: it is a list of flags, not a shell command.
if [ -n "${BASAL_EXTRA_ARGS:-}" ]; then
  read -r -a extra <<< "${BASAL_EXTRA_ARGS}"
  args+=("${extra[@]}")
fi

echo "basal-worker: starting basal-serve ${args[*]}" >&2
basal-serve "${args[@]}" &

# The handler is the container's foreground process and the only thing RunPod
# talks to. `exec` keeps the exit code and the signals honest.
exec python /app/handler.py
