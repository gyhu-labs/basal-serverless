"""RunPod serverless handler for basal: one batch in, ordered answers out.

The image speaks RunPod's own protocol (``POST /run``, ``/runsync``) and talks to
basal's System One API on container loopback — basal has no authentication, so
``127.0.0.1`` *is* the security boundary.

What this module does: validate each element, forward it once, translate the
answer into the gateway's shape. What it deliberately does not do: retry (the
gateway owns retries), apply confidence thresholds or bands (those live in the
caller), build prompts, keep a key, or log content. The image is
generic — no endpoint id, no project data, no calibration file.

Input, one RunPod job::

    {"input": {"requests": [
        {"state": "...",
         "lang": "pl",
         "questions": {"dept": "Do którego działu?"},
         "rubrics": {"dept": {"cards": "Reklamacje kart", "online": "Bankowość"}},
         "media": [{"type": "image", "data": "<base64>"}]},
        ...]}}

Output, either the batch answered::

    {"answers": [{"questions": {"dept": {"label": "cards",
                                         "scores": {"cards": 0.93, "online": 0.07},
                                         "confidence": 0.93,
                                         "abstained": false}}}]}

or, when the whole batch cannot be served::

    {"error": {"code": "worker_unhealthy", "message": "..."}}

``answers[i]`` belongs to ``requests[i]``: order is preserved. An element that
cannot be served gets ``{"error": {"code": ..., "message": ...}}`` in its own
slot, so one malformed element never costs the rest of the batch.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger("basal-worker")

#: Stable error codes. The gateway keys its fallback ladder on these strings,
#: so they are API: rename nothing here without a coordinated caller change.
CODE_INVALID_BATCH = "invalid_batch"
CODE_INVALID_REQUEST = "invalid_request"
CODE_ENGINE_OOM = "engine_oom"
CODE_ENGINE_ERROR = "engine_error"
CODE_WORKER_UNHEALTHY = "worker_unhealthy"

#: basal's own limits (docs/API.md): at most 8 media items of 20 MB each.
_MAX_MEDIA_ITEMS = 8
_MAX_MEDIA_BYTES = 20 * 1024 * 1024
_MEDIA_TYPES = {"image", "audio"}
_DETAILS = {"default", "high"}
_LANGS = {"pl", "en"}

#: An engine error message is echoed to the caller; keep it short enough to sit
#: in a webhook body and to never turn one line of a log into a paragraph.
_MESSAGE_CHARS = 500

#: Engine URLs already seen healthy in this container. A cold start pays the
#: 180 s health gate once, not once per job.
_READY: set[str] = set()


class Malformed(Exception):
    """One element of ``input.requests`` is not a valid basal request."""


class Unhealthy(Exception):
    """The engine did not report ``status: ok`` inside the boot budget."""


@dataclass(frozen=True)
class Settings:
    """Where the engine is and how long to wait for it (env, read per batch)."""

    engine_url: str = "http://127.0.0.1:8001"
    health_timeout_s: float = 180.0
    health_poll_s: float = 1.0
    request_timeout_s: float = 300.0
    #: The whole job must fit RunPod's `executionTimeout` (300 s), or the platform
    #: kills the container and the gateway gets a timeout instead of answers. The
    #: budget is 10 s short of it so the last element returns an error the caller
    #: can act on rather than losing the batch.
    batch_timeout_s: float = 290.0

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            engine_url=os.environ.get("BASAL_URL", cls.engine_url).rstrip("/"),
            health_timeout_s=_env_float("BASAL_HEALTH_TIMEOUT_S", cls.health_timeout_s),
            health_poll_s=_env_float("BASAL_HEALTH_POLL_S", cls.health_poll_s),
            request_timeout_s=_env_float("BASAL_REQUEST_TIMEOUT_S", cls.request_timeout_s),
            batch_timeout_s=_env_float("BASAL_BATCH_TIMEOUT_S", cls.batch_timeout_s),
        )


def handler(job: Mapping[str, Any] | None) -> dict[str, Any]:
    """Serve one RunPod job: every element in order, or one batch-level error."""
    started = time.monotonic()
    requests = _requests_of(job)
    if requests is None:
        LOG.warning("batch rejected code=%s", CODE_INVALID_BATCH)
        return _error(CODE_INVALID_BATCH, "input.requests must be a non-empty list")

    settings = Settings.from_env()
    try:
        _wait_until_ready(settings)
    except Unhealthy as exc:
        LOG.warning(
            "batch failed code=%s n_requests=%d latency_ms=%.0f",
            CODE_WORKER_UNHEALTHY,
            len(requests),
            _ms(started),
        )
        return _error(CODE_WORKER_UNHEALTHY, str(exc))

    deadline = started + settings.batch_timeout_s
    answers = [_answer_one(settings, element, deadline) for element in requests]
    failed = sum(1 for answer in answers if "error" in answer)
    LOG.info(
        "batch done n_requests=%d n_failed=%d latency_ms=%.0f",
        len(requests),
        failed,
        _ms(started),
    )
    return {"answers": answers}


def _answer_one(settings: Settings, element: Any, deadline: float) -> dict[str, Any]:
    """Forward one element to the engine and translate its answer."""
    started = time.monotonic()
    try:
        body = _basal_request(element)
    except Malformed as exc:
        LOG.warning("item failed code=%s latency_ms=%.0f", CODE_INVALID_REQUEST, _ms(started))
        return _error(CODE_INVALID_REQUEST, str(exc))

    n_questions = len(body["questions"])
    media_bytes = sum(len(item["data"]) * 3 // 4 for item in body.get("media", ()))

    # An element the budget cannot fit is reported, not attempted: RunPod would
    # otherwise kill the whole job and the caller would lose the answers it has.
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        LOG.warning(
            "item failed code=%s reason=budget latency_ms=%.0f", CODE_ENGINE_ERROR, _ms(started)
        )
        return _error(CODE_ENGINE_ERROR, "batch time budget exhausted before this element was sent")

    status, payload = _call(
        settings, "/v1/systemone", body, min(settings.request_timeout_s, remaining)
    )
    if status != 200:
        code = _code_for(status)
        # Log the code and the counts only: the engine's message may quote the
        # request, and a log line must never carry the state or the media.
        LOG.warning(
            "item failed code=%s n_questions=%d latency_ms=%.0f", code, n_questions, _ms(started)
        )
        return _error(code, _engine_message(status, payload))

    raw_answers = payload.get("answers") if isinstance(payload, Mapping) else None
    if not isinstance(raw_answers, Mapping):
        LOG.warning(
            "item failed code=%s n_questions=%d latency_ms=%.0f",
            CODE_ENGINE_ERROR,
            n_questions,
            _ms(started),
        )
        return _error(CODE_ENGINE_ERROR, "engine returned no answers")

    result: dict[str, Any] = {}
    for name, question in body["questions"].items():
        answer = raw_answers.get(name)
        if not isinstance(answer, Mapping):
            LOG.warning(
                "item failed code=%s n_questions=%d latency_ms=%.0f",
                CODE_ENGINE_ERROR,
                n_questions,
                _ms(started),
            )
            return _error(CODE_ENGINE_ERROR, f"engine returned no answer for question {name!r}")
        result[name] = _answer_shape(question["type"], answer)

    LOG.info(
        "item ok n_questions=%d media_bytes=%d latency_ms=%.0f",
        n_questions,
        media_bytes,
        _ms(started),
    )
    return {"questions": result}


def _answer_shape(qtype: str, answer: Mapping[str, Any]) -> dict[str, Any]:
    """One basal answer as ``{label, scores, confidence, abstained}``.

    ``abstained`` is basal's own deferral (an ``act``/``tool`` question with
    ``max_error``) or the absence of a decisive label — never a caller-side band: the
    caller owns the thresholds and applies them to ``scores``.
    """
    probabilities = answer.get("probabilities")
    scores = (
        {str(key): float(value) for key, value in probabilities.items()}
        if isinstance(probabilities, Mapping)
        else {}
    )
    confidence = answer.get("confidence")
    label = _label(qtype, answer, scores)
    deferred = bool(answer.get("deferred"))
    abstained = deferred or (label is None and qtype != "multi")
    return {
        "label": label,
        "scores": scores,
        "confidence": float(confidence) if isinstance(confidence, int | float) else None,
        "abstained": abstained,
    }


def _label(qtype: str, answer: Mapping[str, Any], scores: Mapping[str, float]) -> str | None:
    if answer.get("deferred"):
        return None
    if qtype == "choice":
        return _text(answer.get("choice"))
    if qtype == "act":
        return _text(answer.get("action"))
    if qtype == "tool":
        return _text(answer.get("tool"))
    if qtype == "noul":
        probability = answer.get("noul")
        if not isinstance(probability, int | float):
            return None
        return "true" if float(probability) >= 0.5 else "false"
    if qtype == "score":
        # `score` is the expected level; the label a caller can compare is the
        # most probable level key, which is what `probabilities` is keyed by.
        return max(scores, key=lambda key: scores[key]) if scores else None
    # `multi` answers a *set*; its labels are in `selected` and never collapse
    # into one, so the caller reads `scores`.
    return None


def _basal_request(element: Any) -> dict[str, Any]:
    """Validate one element and render it as a basal ``/v1/systemone`` body."""
    if not isinstance(element, Mapping):
        raise Malformed("request must be a JSON object")
    state = element.get("state", "")
    if not isinstance(state, str):
        raise Malformed("state must be a string")
    questions = element.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise Malformed("questions must be a non-empty object")
    rubrics = element.get("rubrics")
    if rubrics is None:
        rubrics = {}
    elif not isinstance(rubrics, Mapping):
        raise Malformed("rubrics must be an object or null")
    media = _media(element.get("media"))
    if not state.strip() and not media:
        raise Malformed("state is empty and no media was given")

    body: dict[str, Any] = {
        "state": state,
        "questions": {
            str(name): _question(str(name), spec, rubrics.get(name))
            for name, spec in questions.items()
        },
    }
    lang = element.get("lang", "pl")
    if lang not in _LANGS:
        raise Malformed('lang must be "pl" or "en"')
    body["lang"] = lang
    if media:
        body["media"] = media
    detail = element.get("detail")
    if detail is not None:
        if detail not in _DETAILS:
            raise Malformed('detail must be "default" or "high"')
        body["detail"] = detail
    return body


def _question(name: str, spec: Any, rubric: Any) -> dict[str, Any]:
    """A basal question object, from either a basal object or text + rubric."""
    if isinstance(spec, Mapping):
        question = dict(spec)
        if not question.get("type"):
            raise Malformed(f"question {name!r} has no type")
        return question
    if not isinstance(spec, str) or not spec.strip():
        raise Malformed(f"question {name!r} must be text or a question object")
    text = spec.strip()
    if rubric is None:
        # No option set: the only answerable shape is yes/no.
        return {"type": "noul", "instructions": text}
    if isinstance(rubric, str):
        if not rubric.strip():
            return {"type": "noul", "instructions": text}
        return {"type": "noul", "instructions": f"{text}\n{rubric.strip()}"}
    if isinstance(rubric, Mapping | list | tuple):
        options = dict(rubric) if isinstance(rubric, Mapping) else list(rubric)
        if len(options) < 2:
            raise Malformed(f"rubric for question {name!r} needs at least two options")
        if len(options) > 255:
            raise Malformed(f"rubric for question {name!r} has more than 255 options")
        return {"type": "choice", "instructions": text, "criteria": options}
    raise Malformed(f"rubric for question {name!r} must be text, a list or an object")


def _media(value: Any) -> list[dict[str, str]]:
    """Validated media items — the bytes are checked, never logged."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise Malformed("media must be a list or null")
    if len(value) > _MAX_MEDIA_ITEMS:
        raise Malformed(f"media carries {len(value)} items, at most {_MAX_MEDIA_ITEMS} are allowed")
    items: list[dict[str, str]] = []
    for position, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise Malformed(f"media[{position}] must be an object")
        kind = item.get("type")
        if kind not in _MEDIA_TYPES:
            raise Malformed(f'media[{position}].type must be "image" or "audio"')
        data = item.get("data")
        if not isinstance(data, str) or not data:
            raise Malformed(f"media[{position}].data must be a non-empty base64 string")
        # `base64` on the command line wraps at 76 columns; the engine takes one
        # line, so the whitespace is stripped (and the stripped form forwarded).
        data = "".join(data.split())
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise Malformed(f"media[{position}].data is not base64") from exc
        if not decoded:
            raise Malformed(f"media[{position}].data decodes to nothing")
        if len(decoded) > _MAX_MEDIA_BYTES:
            raise Malformed(f"media[{position}] is larger than {_MAX_MEDIA_BYTES} bytes")
        items.append({"type": kind, "data": data})
    return items


def _wait_until_ready(settings: Settings) -> None:
    """Poll ``GET /health`` until ``status: ok``, or raise inside the budget."""
    if settings.engine_url in _READY:
        return
    started = time.monotonic()
    last = "no response"
    while True:
        status, body = _call(settings, "/health", None, min(5.0, settings.health_timeout_s))
        if status == 200 and isinstance(body, Mapping) and body.get("status") == "ok":
            LOG.info(
                "engine ready running=%s queued=%s wait_ms=%.0f",
                body.get("running"),
                body.get("queued"),
                _ms(started),
            )
            _READY.add(settings.engine_url)
            return
        last = f"status={status}" + (
            f" error={_engine_message(status, body)}"
            if isinstance(body, Mapping) and body.get("error")
            else ""
        )
        elapsed = time.monotonic() - started
        if elapsed >= settings.health_timeout_s:
            raise Unhealthy(
                f"engine at {settings.engine_url} did not become healthy within "
                f"{settings.health_timeout_s:.0f}s ({last})"
            )
        time.sleep(min(settings.health_poll_s, settings.health_timeout_s - elapsed))


def _call(
    settings: Settings, path: str, payload: Mapping[str, Any] | None, timeout: float
) -> tuple[int | None, Any]:
    """One HTTP call to the engine: (status, decoded body). No retries."""
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{settings.engine_url}{path}",
        data=data,
        headers={"content-type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), _json(response.read())
    except urllib.error.HTTPError as exc:
        return int(exc.code), _json(exc.read())
    except TimeoutError:
        # The engine is up but slow: its readiness is still true.
        return None, {"error": f"engine did not answer within {timeout:.0f}s"}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        # A refused or broken connection means the cached readiness is stale, so
        # the next job pays the health gate again instead of guessing.
        _READY.discard(settings.engine_url)
        return None, {"error": f"engine unreachable ({exc.__class__.__name__})"}


def _json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode() or "null")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _code_for(status: int | None) -> str:
    """basal's status codes as the gateway's error codes (docs/API.md)."""
    if status == 422:
        return CODE_INVALID_REQUEST
    if status == 503:
        return CODE_ENGINE_OOM
    return CODE_ENGINE_ERROR


def _engine_message(status: int | None, payload: Any) -> str:
    if isinstance(payload, Mapping):
        message = payload.get("error")
        if isinstance(message, str) and message:
            return message[:_MESSAGE_CHARS]
    if status is None:
        return "engine unreachable"
    return f"engine returned HTTP {status}"


def _requests_of(job: Mapping[str, Any] | None) -> list[Any] | None:
    if not isinstance(job, Mapping):
        return None
    payload = job.get("input")
    if not isinstance(payload, Mapping):
        return None
    requests = payload.get("requests")
    if not isinstance(requests, list) or not requests:
        return None
    return requests


def _error(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _ms(started: float) -> float:
    return (time.monotonic() - started) * 1000.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def main() -> None:  # pragma: no cover - starts the RunPod SDK
    logging.basicConfig(
        level=os.environ.get("BASAL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    import runpod

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":  # pragma: no cover
    main()
