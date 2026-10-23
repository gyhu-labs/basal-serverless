# basal-serverless

A generic RunPod serverless image for the [basal](https://github.com/rkinas/basal)
decision engine. It carries the engine, a handler, and nothing else: no API key,
no endpoint id, no app or webhook URL, no project data, no prompts, no
calibration file. It forwards payloads and returns structure — everything that
makes a payload *ours* lives in the service that calls the endpoint, never here.

The image speaks RunPod's own protocol (`POST /run`, `/runsync`) through
`runpod.serverless.start`, and talks to `basal-serve` on `127.0.0.1:8001`.
basal has no authentication, so container loopback **is** the security boundary:
the engine binds loopback only and nothing else in the image binds a port.

```
RunPod queue ──▶ handler.py ──▶ 127.0.0.1:8001 /v1/systemone
                 (validate,      (basal-serve, one call per element,
                  translate)      all questions of that element in one pass)
```

## Build and run

```bash
docker build -t basal-worker:test basal-worker/          # text-only, no weights
docker build -t basal-worker:test --build-arg FETCH_MODEL=1 basal-worker/
```

```bash
# CPU-only smoke of the container contract (the engine itself needs a GPU):
docker run --rm -e BASAL_MODEL=Remek/basal-omni-1.0-mini basal-worker:test \
    python -c "import handler; print(handler.Settings.from_env())"
```

Weights are **not** in the image by default: `FETCH_MODEL=1` bakes the model
into `/models/hf` at build time, which is what a release image should ship (a
serverless container's own filesystem is ephemeral). With the default, the
engine downloads on first boot and a cold start pays for it — fine for a test
endpoint, not for a release.

## Environment

| variable | default | meaning |
|---|---|---|
| `BASAL_MODEL` | `Remek/basal-omni-1.0-mini` | engine model (repo id or a path under `/models`) |
| `BASAL_MODE` | `fast-nocompile` | text-only CUDA-graph mode; `fast` compiles for minutes on first start |
| `BASAL_MEDIA` | `off` | `on` switches the engine to `--mode eager --media on` (images/audio) |
| `BASAL_PORT` | `8001` | engine port, always bound on `127.0.0.1` (8000 belongs to the RunPod SDK) |
| `BASAL_CALIBRATION` | `/calibration/CALIBRATION.custom.json` | used **if the file is mounted**, with `--strict-calibration` |
| `BASAL_EXTRA_ARGS` | *(empty)* | extra `basal-serve` flags, e.g. `--max-len 16384` |
| `BASAL_URL` | `http://127.0.0.1:8001` | where the handler looks for the engine |
| `BASAL_HEALTH_TIMEOUT_S` | `180` | boot gate; past it every batch answers `worker_unhealthy` |
| `BASAL_REQUEST_TIMEOUT_S` | `300` | per-element engine timeout (RunPod's `executionTimeout` is 300 s) |
| `BASAL_BATCH_TIMEOUT_S` | `290` | the whole job, 10 s short of `executionTimeout`: an element the budget cannot fit is reported, never attempted, so a slow one cannot get the job killed and lose the answers already computed |
| `BASAL_LOG_LEVEL` | `INFO` | handler log level |

The media path is off by default on purpose: images go to the vision model
first, and `--media on` is an eager-mode feature (text and media do not share a
CUDA-graph configuration). Flip `BASAL_MEDIA=on` only for an endpoint that is
meant to read photos.

A state longer than the engine's context is **refused** (`--context-overflow
refuse`), not truncated: a silently cut state yields a confident wrong answer,
while a 422 lets the caller split the request. Media may be base64 with line
breaks (`base64 photo.jpg` wraps at 76 columns); it is forwarded compacted.

A job has to finish inside RunPod's `executionTimeout` (300 s) or the platform
kills the container and the caller loses every answer in it. The handler keeps
its own budget (`BASAL_BATCH_TIMEOUT_S`, 290 s): each element is sent with the
time the budget has left, and an element that no longer fits gets an
`engine_error` saying so — the answers already computed come back.

## Request

One RunPod job, `input.requests[]` in the order the answers must come back:

```json
{
  "input": {
    "requests": [
      {
        "state": "Zgłoszenie z oddziału: od rana nie działa terminal, kolejka rośnie.",
        "lang": "pl",
        "questions": {
          "urgent": "Czy sprawa wymaga reakcji dziś?",
          "dept": "Który zespół powinien ją przejąć?"
        },
        "rubrics": {
          "dept": {
            "payments": "Terminale i płatności",
            "facilities": "Usterki lokalu",
            "other": "Wszystko inne"
          }
        },
        "media": [{"type": "image", "data": "<base64 JPEG>"}]
      }
    ]
  }
}
```

* `state` — the text to decide over (may be empty when media is present).
* `questions` — name → either question text (see `rubrics`) or a basal question
  object (`{"type": "choice" | "noul" | "score" | "multi" | "act" | "tool",
  "instructions": ..., "criteria": ...}`), which is forwarded unchanged.
* `rubrics` — optional, per question: an **object or list of options** makes the
  question a `choice`; a **string** is appended to the question text as extra
  guidance; **absent** means yes/no (`noul`).
* `media` — optional, `image` or `audio`, base64, at most 8 items of 20 MB each,
  and only meaningful when the engine runs with `BASAL_MEDIA=on`.
* `detail` — optional, `default` (280 tokens) or `high` (1,120 tokens) per image.

## Response

```json
{
  "answers": [
    {
      "questions": {
        "urgent": {"label": "true", "scores": {"true": 0.94, "false": 0.06},
                   "confidence": 0.94, "abstained": false},
        "dept": {"label": "payments", "scores": {"payments": 0.81, "facilities": 0.12, "other": 0.07},
                 "confidence": 0.81, "abstained": false}
      }
    }
  ]
}
```

`answers[i]` belongs to `requests[i]`. `scores` are basal's calibrated
probabilities; `label` is the chosen option (`true`/`false` for yes/no, the
most probable level for a score question, `null` for a set question — read
`scores`); `abstained` is basal's own deferral or the absence of a decision.
**No thresholds or bands are applied here**: the caller owns them.

Failure modes, per element or for the whole batch:

| code | scope | meaning |
|---|---|---|
| `invalid_batch` | batch | `input.requests` is missing, empty or not a list |
| `worker_unhealthy` | batch | the engine never answered `/health` with `status: ok` inside the boot budget |
| `invalid_request` | element | the element is malformed, or the engine refused it (422) |
| `engine_oom` | element | the engine ran out of GPU memory (503) — a later retry may succeed |
| `engine_error` | element | any other engine failure (500, no answer, unreachable) |

A failed element keeps its slot: `{"error": {"code": ..., "message": ...}}`.
One malformed element never costs the rest of the batch, and the handler never
retries — retries, fallbacks and budget rules belong to the calling gateway.

## Logs

Counts and timings only, one line per element and one per batch:
`n_requests`, `n_questions`, `media_bytes`, `latency_ms`, and the error code on
failure. Never the state, never a transcript, never media bytes, and never the
engine's error message (which may quote the request) — that one goes to the
caller, in the response.

## Tests

```bash
env -u PYTHONPATH uv run pytest basal-worker/tests -q
```

The suite runs against a fake engine on loopback: no GPU, no weights, no
network. It covers batch ordering, one-call-per-element (several questions share
a state), per-element error isolation, the 180 s health gate, the 422/503/500
mapping, 1 MB media pass-through, the logging rule, and the generic-image rules
(no key-shaped string, no foreign URL, the pinned engine version).

## Verified on the built image

Measured on the image this directory builds (`basal-worker:test`, amd64, no GPU,
`FETCH_MODEL=0`), so a reviewer can tell what has been checked and what has not:

| what | value |
|---|---|
| size | 4.43 GB as `docker image inspect` reports it (what a registry transfers), ~8.2 GB on disk — the build asserts the latter against a 9 GB budget |
| cold build | 110 s with the base image cached, 235 s from scratch |
| container start → handler ready (`import handler`) | 0.76 s |
| handler import alone | 40 ms |
| `POST /runsync`, 3 requests × 2 questions | 3 answers, input order preserved |
| one malformed element among two good ones | `invalid_request` in that slot, neighbours answered |
| 1 MB base64 image | carried through unchanged, `media_bytes=1048581` logged |
| container logs | counts and timings only; no state, no media |
| **model load (≤ 30 s budget)** | **not measured — needs a GPU**; measure the real cold start on your endpoint |

```bash
# The same end-to-end check, against any engine on the host's port 18080:
# the RunPod SDK's own API server (`--rp_serve_api`) is what exposes /runsync.
docker run --rm -p 18000:8000 --add-host host.docker.internal:host-gateway \
    -e BASAL_URL=http://host.docker.internal:18080 \
    --entrypoint python basal-worker:test \
    /app/handler.py --rp_serve_api --rp_api_host 0.0.0.0 --rp_api_port 8000
curl -s localhost:18000/runsync -H 'content-type: application/json' \
    -d '{"input": {"requests": [{"state": "...", "questions": {"q": "?"}}]}}'
```

## Licence

Everything in this repository is licensed under the Apache License, Version 2.0
(`LICENSE`). The bundled engine and model weights are Apache-2.0 third-party
work: see `NOTICE` for their attribution, including the Gemma 4 models the
weights are derived from.
