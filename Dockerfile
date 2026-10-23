# syntax=docker/dockerfile:1.28
#
# The generic basal worker image: a decision engine behind RunPod's protocol.
#
#   input   RunPod's queue (`POST /run`, `/runsync`) via `handler.py`
#   engine  basal-serve on 127.0.0.1:8001 only — basal has no authentication, so
#           container loopback IS the security boundary
#
# The image is generic by contract: no API key, no endpoint id, no
# app or webhook URL, no project data, no prompts, no calibration file. It
# forwards payloads and returns structure; `basal-worker/tests` and the
# `basal-worker` workflow prove that on every change.
#
# Base image: the exact pair basal's install guide checks — torch 2.11.0 built
# for CUDA 12.8, with the matching torchvision already present. Building it from
# a bare `nvidia/cuda` base would download the same ~3 GB of wheels and risk a
# torchvision/torch mismatch, which breaks `transformers` at import.
#
# The media path (basal-omni image + audio) is installed but *off* by default:
# the engine boots `--mode fast-nocompile` (text only). Set `BASAL_MEDIA=on` to
# serve `--mode eager --media on` instead (delta 2026-10-22: images go to the
# vision model first, so the worker is text-only unless asked otherwise).
#
# Weights: `FETCH_MODEL=0` (default) downloads on first boot / relies on the
# platform's cached model; `FETCH_MODEL=1` bakes them into /models at build
# time, which is what a release image should ship (RunPod's container disk is
# ephemeral). The build context never touches the weights either way.
#
# LAYER BUDGET — do not split what follows into more `RUN`s. Every `RUN` on top
# of this 4.3 GB base copies the whole base into a new overlay snapshot, so a
# helpful-looking extra step (a second pip call, a separate version check, a
# separate weights stage) costs another 4.3 GB of build disk. On the CI runner
# that is the difference between a green build and `no space left on device` —
# which is exactly how the first version of this file failed. `COPY --link` is
# used for the same reason: it adds a layer without materialising the base.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

FROM ${PYTORCH_IMAGE} AS image

# basal 2.3.5 is the pinned engine. 2.3.4 captures basal-omni's CUDA graphs
# largest first (basal-omni-1.0-mini 27.8 -> 18.9 GB on one H100); with 2.3.1 the
# mini model does not start on a 24 GB card, so the worker never got healthy.
# The weights did not change. The version is a build arg so the pin is visible
# and testable, but no caller is expected to move it.
ARG BASAL_VERSION=2.3.5
ARG RUNPOD_VERSION=1.12.0
ARG FETCH_MODEL=0
ARG BASAL_MODEL=Remek/basal-omni-1.0-mini
#: On-disk rootfs ceiling, measured below with `du` (the CUDA runtime base alone
#: is ~7.8 GB on disk, which a registry reports as 4.3 GB of layers, and the
#: engine stack adds ~0.4 GB). 9 GB catches an accidental second torch install or
#: a model that arrived without `FETCH_MODEL=1` being asked for.
ARG SIZE_BUDGET_GB=9

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    BASAL_URL=http://127.0.0.1:8001 \
    BASAL_LOG_LEVEL=INFO \
    VIRTUAL_ENV=/opt/basal \
    PATH=/opt/basal/bin:${PATH}

# Not `/tmp`: `COPY --link` replaces the directory's mode, and apt's sandboxed
# key handling then cannot write its temp file in a /tmp that lost the sticky
# 1777 bit (E: "The repository ... is not signed").
COPY --link requirements.txt /opt/constraints.txt
COPY --link entrypoint.sh handler.py healthcheck.py /app/

# ONE layer: the interpreter, the engine, the pin proof, the optional weights and
# the unprivileged user. See LAYER BUDGET above before adding anything here.
#
# The engine gets its own venv, because the base image's Python is PEP-668
# externally managed and already carries torch 2.11.0+cu128 and torchvision
# 0.26.0 in dist-packages: the venv is created with `--system-site-packages` and
# *inherits* those two wheels instead of resolving them again, so
# `requirements.txt` — applied as constraints, not requirements — has nothing to
# disagree with. `python3.12-venv` is the one package the base lacks for
# `ensurepip`.
#
# `spin` is PyTorch's own developer CLI, pinned in the base to `click<8.4`; the
# engine's `huggingface_hub` requires `click>=8.4.2`, so the two cannot coexist
# and `pip check` would otherwise fail on a conflict nothing in this image uses —
# the worker never runs `spin`, so that one dev tool goes.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends python3.12-venv; \
    rm -rf /var/lib/apt/lists/*; \
    python -m venv --system-site-packages "${VIRTUAL_ENV}"; \
    /usr/bin/python -m pip uninstall --yes --no-input --break-system-packages spin; \
    pip install --no-cache-dir "runpod==${RUNPOD_VERSION}"; \
    pip install --no-cache-dir -c /opt/constraints.txt \
      "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v${BASAL_VERSION}.tar.gz"; \
    pip check; \
    test "$(python -c 'import basal; print(basal.__version__)')" = "${BASAL_VERSION}"; \
    basal-serve --help > /dev/null; \
    mkdir -p /models/hf /calibration; \
    if [ "${FETCH_MODEL}" = "1" ]; then \
      python -c "from huggingface_hub import snapshot_download as d; d('${BASAL_MODEL}')"; \
    fi; \
    chmod 0755 /app/entrypoint.sh /app/handler.py /app/healthcheck.py; \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin worker; \
    chown -R worker:worker /models /app /home/worker; \
    rm -f /opt/constraints.txt; \
    test ! -e /calibration/CALIBRATION.custom.json; \
    ! grep -REn "rp_[A-Za-z0-9]{14,}|api\.runpod\.ai/v2/" /app; \
    test -z "$(env | grep -Ei '^(RUNPOD_API_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY)=' || true)"; \
    rootfs="$(du -sb --one-file-system --exclude=/models / | cut -f1)"; \
    echo "basal-worker: rootfs ${rootfs} bytes (budget $(( ${SIZE_BUDGET_GB} * 1000000000 )) bytes)"; \
    if [ "${FETCH_MODEL}" = "0" ]; then \
      test "${rootfs}" -lt "$(( ${SIZE_BUDGET_GB} * 1000000000 ))"; \
    fi

ENV BASAL_MODEL=${BASAL_MODEL} \
    BASAL_MODE=fast-nocompile \
    BASAL_MEDIA=off \
    BASAL_PORT=8001 \
    BASAL_CALIBRATION=/calibration/CALIBRATION.custom.json \
    HOME=/home/worker

# A non-root worker with exactly two writable places: the weight cache and its
# own home (triton / torch compile caches). /calibration is read-only by design:
# the file arrives mounted at deploy time, never baked in.
USER worker
WORKDIR /app

# Two ports, two different things:
#   8000  the RunPod SDK's own API server (`--rp_serve_api`, `/runsync`, `/run`);
#         in production the SDK long-polls instead, so nothing binds it — but it
#         is the port the platform's HTTP surface expects, so the engine never
#         takes it
#   8001  basal-serve, loopback only, the handler's only client
EXPOSE 8000

# `running`/`queued` come from basal's /health (2.3.1+). `start-period` covers a
# cold model load; the handler's own gate is the 180 s one the gateway sees.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD ["python", "/app/healthcheck.py"]

ENTRYPOINT ["/app/entrypoint.sh"]
