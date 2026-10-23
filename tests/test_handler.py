"""Synthetic tests for the basal worker handler — no GPU, no model, no network.

Every test runs against a fake engine on an ephemeral loopback port and fake
payloads only: the suite is the worker's contract, not the model's quality. It
therefore lives beside the image (run with ``uv run pytest basal-worker/tests``)
rather than in the product's ``tests/`` lane, which must not depend on a
container.

The fake engine answers the two paths the handler uses — ``GET /health`` and
``POST /v1/systemone`` — with basal 2.3.1's shapes (``docs/API.md``).
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

WORKER = Path(__file__).resolve().parents[1]
if str(WORKER) not in sys.path:
    sys.path.insert(0, str(WORKER))

import handler as worker  # noqa: E402

Responder = Callable[[str, Any], "tuple[int, dict[str, Any]]"]

#: A state that must never reach a log line, a file or an image layer.
SENTINEL = "SENTINEL-STATE-7f3a"


class FakeEngine:
    """A loopback engine: one responder, every call recorded."""

    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self.calls: list[tuple[str, Any]] = []
        self._closed = False
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:  # keep pytest's output clean
                return

            def _serve(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                raw = self.rfile.read(length) if length else b""
                body = json.loads(raw) if raw else None
                outer.calls.append((self.path, body))
                status, payload = outer.responder(self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = _serve
            do_POST = _serve

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def posted_bodies(self) -> list[Any]:
        return [body for path, body in self.calls if path == "/v1/systemone"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


@pytest.fixture(autouse=True)
def _fresh_readiness() -> Iterator[None]:
    """The health gate is cached per container; each test starts cold."""
    worker._READY.clear()
    yield
    worker._READY.clear()


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., FakeEngine]]:
    """Start a fake engine and point the handler at it."""
    started: list[FakeEngine] = []

    def start(responder: Responder, *, health_timeout: float = 5.0) -> FakeEngine:
        fake = FakeEngine(responder)
        started.append(fake)
        monkeypatch.setenv("BASAL_URL", fake.url)
        monkeypatch.setenv("BASAL_HEALTH_TIMEOUT_S", str(health_timeout))
        monkeypatch.setenv("BASAL_HEALTH_POLL_S", "0.05")
        monkeypatch.setenv("BASAL_REQUEST_TIMEOUT_S", "10")
        return fake

    yield start
    for fake in started:
        fake.close()


def healthy(path: str, body: Any) -> tuple[int, dict[str, Any]]:
    """basal 2.3.1's happy path: health idle, one answer per question."""
    if path == "/health":
        return 200, {"status": "ok", "running": 0, "queued": 0}
    if path != "/v1/systemone":
        return 404, {"error": "not found"}
    answers: dict[str, Any] = {}
    for name, question in body["questions"].items():
        if question["type"] == "choice":
            keys = list(question.get("criteria") or {})
            answers[name] = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": {key: 0.9 if key == keys[0] else 0.1 for key in keys},
                "confidence": 0.9,
            }
        else:
            answers[name] = {
                "type": "noul",
                "noul": 0.75,
                "probabilities": {"true": 0.75, "false": 0.25},
                "confidence": 0.75,
            }
    return 200, {"model": "fake", "answers": answers, "usage": {"questions": len(answers)}}


def numbered(path: str, body: Any) -> tuple[int, dict[str, Any]]:
    """Answer each request with a value derived from its state (order probe)."""
    if path == "/health":
        return 200, {"status": "ok", "running": 0, "queued": 0}
    index = int(body["state"].rsplit(" ", 1)[-1])
    noul = 0.1 + 0.45 * index
    answer = {
        "type": "noul",
        "noul": noul,
        "probabilities": {"true": noul, "false": round(1 - noul, 4)},
        "confidence": 0.11 * (index + 1),
    }
    return 200, {"answers": dict.fromkeys(body["questions"], answer)}


def one_question(state: str = SENTINEL) -> dict[str, Any]:
    return {"state": state, "lang": "pl", "questions": {"urgent": "Czy sprawa jest pilna?"}}


# --- the batch contract (EARS 2) ---------------------------------------------


def test_single_question(engine: Callable[..., FakeEngine]) -> None:
    """One question, one answer, in the gateway's shape."""
    engine(healthy)

    result = worker.handler({"input": {"requests": [one_question()]}})

    assert "error" not in result
    (answer,) = result["answers"]
    assert set(answer["questions"]) == {"urgent"}
    urgent = answer["questions"]["urgent"]
    assert urgent == {
        "label": "true",
        "scores": {"true": 0.75, "false": 0.25},
        "confidence": 0.75,
        "abstained": False,
    }


def test_five_questions_share_one_call(engine: Callable[..., FakeEngine]) -> None:
    """SOAM: five questions about one state are ONE engine request."""
    fake = engine(healthy)
    questions = {f"q{index}": f"Pytanie numer {index}?" for index in range(5)}

    result = worker.handler({"input": {"requests": [{"state": SENTINEL, "questions": questions}]}})

    assert len(fake.posted_bodies()) == 1
    assert len(fake.posted_bodies()[0]["questions"]) == 5
    assert set(result["answers"][0]["questions"]) == set(questions)


def test_batch_order_preserved(engine: Callable[..., FakeEngine]) -> None:
    """EARS 2: three requests, two questions each, answers in input order."""
    fake = engine(numbered)
    requests = [
        {
            "state": f"sprawa {index}",
            "questions": {"urgent": "Czy pilne?", "kind": "Jaki rodzaj?"},
        }
        for index in range(3)
    ]

    result = worker.handler({"input": {"requests": requests}})

    answers = result["answers"]
    assert len(answers) == 3
    assert all(set(answer["questions"]) == {"urgent", "kind"} for answer in answers)
    assert [answer["questions"]["urgent"]["confidence"] for answer in answers] == pytest.approx(
        [0.11, 0.22, 0.33]
    )
    assert [answer["questions"]["urgent"]["label"] for answer in answers] == [
        "false",
        "true",
        "true",
    ]
    assert len(fake.posted_bodies()) == 3


# --- one bad element never costs the batch (EARS 3) ---------------------------


def test_malformed_element_isolated(engine: Callable[..., FakeEngine]) -> None:
    """EARS 3: the broken element gets an error, its neighbours are answered."""
    fake = engine(healthy)
    requests = [
        one_question("sprawa 1"),
        {"state": "bez pytań", "questions": {}},
        one_question("sprawa 2"),
    ]

    result = worker.handler({"input": {"requests": requests}})

    answers = result["answers"]
    assert [("error" in answer) for answer in answers] == [False, True, False]
    assert answers[1]["error"]["code"] == "invalid_request"
    assert "questions" in answers[0] and "questions" in answers[2]
    # The malformed element is never forwarded: validation happens here.
    assert len(fake.posted_bodies()) == 2


@pytest.mark.parametrize(
    ("element", "fragment"),
    [
        ({"state": "x", "questions": {"q": ""}}, "must be text"),
        ({"state": "x", "questions": {"q": 42}}, "must be text"),
        ({"state": "x", "questions": {"q": "?"}, "lang": "de"}, "lang"),
        ({"state": "", "questions": {"q": "?"}}, "state is empty"),
        ({"state": "x", "questions": {"q": "?"}, "rubrics": []}, "rubrics"),
        ({"state": 7, "questions": {"q": "?"}}, "state must be a string"),
        ("not an object", "JSON object"),
    ],
)
def test_invalid_elements_are_rejected(
    engine: Callable[..., FakeEngine], element: Any, fragment: str
) -> None:
    fake = engine(healthy)

    result = worker.handler({"input": {"requests": [element]}})

    assert result["answers"][0]["error"]["code"] == "invalid_request"
    assert fragment in result["answers"][0]["error"]["message"]
    assert fake.posted_bodies() == []


def test_media_validation(engine: Callable[..., FakeEngine]) -> None:
    fake = engine(healthy)
    bad_type = {"state": "x", "questions": {"q": "?"}, "media": [{"type": "video", "data": "AA"}]}
    bad_base64 = {"state": "x", "questions": {"q": "?"}, "media": [{"type": "image", "data": "!!"}]}

    result = worker.handler({"input": {"requests": [bad_type, bad_base64]}})

    codes = [answer["error"]["code"] for answer in result["answers"]]
    assert codes == ["invalid_request", "invalid_request"]
    assert fake.posted_bodies() == []


# --- engine failures map to stable codes (EARS 4, delta) ----------------------


def test_engine_oom_is_per_item_and_not_retried(engine: Callable[..., FakeEngine]) -> None:
    """503 is a GPU OOM: per-item error, one call only — the gateway retries."""

    def oom(path: str, body: Any) -> tuple[int, dict[str, Any]]:
        if path == "/health":
            return 200, {"status": "ok", "running": 0, "queued": 0}
        return 503, {"error": "CUDA out of memory"}

    fake = engine(oom)

    result = worker.handler({"input": {"requests": [one_question("a"), one_question("b")]}})

    codes = [answer["error"]["code"] for answer in result["answers"]]
    assert codes == ["engine_oom", "engine_oom"]
    assert "out of memory" in result["answers"][0]["error"]["message"]
    assert len(fake.posted_bodies()) == 2  # one attempt per element, never two


def test_engine_422_is_an_invalid_request(engine: Callable[..., FakeEngine]) -> None:
    def refuse(path: str, body: Any) -> tuple[int, dict[str, Any]]:
        if path == "/health":
            return 200, {"status": "ok", "running": 0, "queued": 0}
        return 422, {"error": "prompt longer than the context"}

    engine(refuse)

    result = worker.handler({"input": {"requests": [one_question()]}})

    assert result["answers"][0]["error"]["code"] == "invalid_request"
    assert "context" in result["answers"][0]["error"]["message"]


def test_engine_500_is_an_engine_error(engine: Callable[..., FakeEngine]) -> None:
    def crash(path: str, body: Any) -> tuple[int, dict[str, Any]]:
        if path == "/health":
            return 200, {"status": "ok", "running": 0, "queued": 0}
        return 500, {"error": "internal"}

    engine(crash)

    result = worker.handler({"input": {"requests": [one_question()]}})

    assert result["answers"][0]["error"]["code"] == "engine_error"


def test_engine_death_is_detected_and_the_gate_is_repaid(
    engine: Callable[..., FakeEngine],
) -> None:
    """A dead connection invalidates the cached readiness.

    The batch that hits the dead engine reports it per item (no retry inside the
    handler), and the *next* job pays the health gate again instead of guessing.
    """
    fake = engine(healthy, health_timeout=0.5)
    assert "error" not in worker.handler({"input": {"requests": [one_question()]}})

    fake.close()  # the port now refuses connections
    after_death = worker.handler({"input": {"requests": [one_question()]}})
    assert after_death["answers"][0]["error"]["code"] == "engine_error"

    next_job = worker.handler({"input": {"requests": [one_question()]}})
    assert next_job["error"]["code"] == "worker_unhealthy"


def test_batch_budget_reports_the_elements_it_could_not_send(
    engine: Callable[..., FakeEngine],
) -> None:
    """RunPod would kill the job at 300 s; the handler reports instead."""
    fake = engine(healthy)
    settings = worker.Settings.from_env()

    # The budget already ran out (the health gate or earlier elements used it):
    # the element is reported, never attempted, so the caller keeps the answers
    # it does have instead of losing the whole job to the platform's timeout.
    result = worker._answer_one(settings, one_question(), time.monotonic() - 1.0)

    assert result["error"]["code"] == "engine_error"
    assert "budget" in result["error"]["message"]
    assert fake.posted_bodies() == []


def test_a_generous_budget_does_not_shorten_a_request(
    engine: Callable[..., FakeEngine],
) -> None:
    """The budget bounds a slow element, and a fast one is unaffected."""
    engine(healthy)
    settings = worker.Settings.from_env()

    result = worker._answer_one(settings, one_question(), time.monotonic() + 60)

    assert "questions" in result


def test_unreachable_engine_fails_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A closed port is a batch-level error, not a crash and not a retry loop."""
    monkeypatch.setenv("BASAL_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("BASAL_HEALTH_TIMEOUT_S", "0.3")
    monkeypatch.setenv("BASAL_HEALTH_POLL_S", "0.05")

    result = worker.handler({"input": {"requests": [one_question()]}})

    assert result["error"]["code"] == "worker_unhealthy"


def test_unhealthy_engine_fails_fast(engine: Callable[..., FakeEngine]) -> None:
    """EARS 4: no healthy engine → one batch-level error, no answers, no hang."""
    fake = engine(lambda path, body: (503, {"error": "model still loading"}), health_timeout=1.0)

    started = time.monotonic()
    result = worker.handler({"input": {"requests": [one_question(), one_question()]}})
    elapsed = time.monotonic() - started

    assert result["error"]["code"] == "worker_unhealthy"
    assert "answers" not in result  # no partial answer for the batch
    assert elapsed < 15  # the real budget is 180 s; the test budget is 1 s
    assert fake.posted_bodies() == []  # no request is sent to a dead engine


def test_health_gate_is_paid_once(engine: Callable[..., FakeEngine]) -> None:
    """A warm container does not re-poll /health on every job."""
    fake = engine(healthy)

    worker.handler({"input": {"requests": [one_question()]}})
    worker.handler({"input": {"requests": [one_question()]}})

    health_calls = [path for path, _ in fake.calls if path == "/health"]
    assert len(health_calls) == 1


def test_invalid_batch(engine: Callable[..., FakeEngine]) -> None:
    fake = engine(healthy)

    for job in (None, {}, {"input": {}}, {"input": {"requests": []}}, {"input": {"requests": {}}}):
        assert worker.handler(job)["error"]["code"] == "invalid_batch"
    assert fake.calls == []


# --- payload translation ------------------------------------------------------


def test_question_without_rubric_is_yes_no(engine: Callable[..., FakeEngine]) -> None:
    fake = engine(healthy)

    worker.handler({"input": {"requests": [one_question()]}})

    question = fake.posted_bodies()[0]["questions"]["urgent"]
    assert question == {"type": "noul", "instructions": "Czy sprawa jest pilna?"}


def test_rubric_object_becomes_choice(engine: Callable[..., FakeEngine]) -> None:
    fake = engine(healthy)

    worker.handler(
        {
            "input": {
                "requests": [
                    {
                        "state": SENTINEL,
                        "questions": {"dept": "Do którego działu?"},
                        "rubrics": {"dept": {"cards": "Reklamacje kart", "online": "Bankowość"}},
                    }
                ]
            }
        }
    )

    question = fake.posted_bodies()[0]["questions"]["dept"]
    assert question["type"] == "choice"
    assert question["criteria"] == {"cards": "Reklamacje kart", "online": "Bankowość"}


def test_basal_question_object_passes_through(engine: Callable[..., FakeEngine]) -> None:
    """The gateway may speak basal's own question shape; the worker keeps it."""
    fake = engine(healthy)
    spec = {"type": "score", "instructions": "Jak pilne?", "criteria": ["niska", "wysoka"]}

    worker.handler({"input": {"requests": [{"state": SENTINEL, "questions": {"urgent": spec}}]}})

    assert fake.posted_bodies()[0]["questions"]["urgent"] == spec


def test_one_megabyte_image_is_forwarded_intact(engine: Callable[..., FakeEngine]) -> None:
    """A 1 MB JPEG-ish payload survives base64 → JSON → engine unchanged."""
    fake = engine(healthy)
    image = b"\xff\xd8\xff\xe0" + b"\x00" * (1024 * 1024)
    encoded = base64.b64encode(image).decode()

    result = worker.handler(
        {
            "input": {
                "requests": [
                    {
                        "state": SENTINEL,
                        "questions": {"amount": "Czy jest kwota?"},
                        "media": [{"type": "image", "data": encoded}],
                    }
                ]
            }
        }
    )

    assert "error" not in result
    body = fake.posted_bodies()[0]
    assert body["media"] == [{"type": "image", "data": encoded}]
    assert len(base64.b64decode(body["media"][0]["data"])) == len(image)


def test_wrapped_base64_is_accepted(engine: Callable[..., FakeEngine]) -> None:
    """`base64 file.jpg` wraps at 76 columns; the engine takes one line."""
    fake = engine(healthy)
    image = b"\xff\xd8\xff" + b"\x02" * 300
    wrapped = "\n".join(
        base64.b64encode(image).decode()[offset : offset + 76]
        for offset in range(0, len(base64.b64encode(image).decode()), 76)
    )

    result = worker.handler(
        {
            "input": {
                "requests": [
                    {
                        "state": SENTINEL,
                        "questions": {"amount": "Czy jest kwota?"},
                        "media": [{"type": "image", "data": wrapped}],
                    }
                ]
            }
        }
    )

    assert "error" not in result
    forwarded = fake.posted_bodies()[0]["media"][0]["data"]
    assert "\n" not in forwarded
    assert base64.b64decode(forwarded) == image


# --- logging carries counts, never content (EARS 5) ---------------------------


def test_no_state_in_logs(
    engine: Callable[..., FakeEngine], caplog: pytest.LogCaptureFixture
) -> None:
    """EARS 5: counts and timings only — no state text, no media bytes."""
    engine(healthy)
    state = f"{SENTINEL} klient nie może się zalogować"
    image = base64.b64encode(b"\xff\xd8\xff" + b"\x01" * 2048).decode()
    caplog.set_level(logging.DEBUG, logger="basal-worker")

    worker.handler(
        {
            "input": {
                "requests": [
                    {
                        "state": state,
                        "questions": {"urgent": "Czy pilne?", "dept": "Który dział?"},
                        "media": [{"type": "image", "data": image}],
                    }
                ]
            }
        }
    )

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert logged, "the handler must log something"
    assert SENTINEL not in logged
    assert "klient nie może" not in logged
    assert image[:32] not in logged
    assert "n_questions=2" in logged
    assert "media_bytes=" in logged
    assert "latency_ms=" in logged


def test_failures_log_codes_not_engine_messages(
    engine: Callable[..., FakeEngine], caplog: pytest.LogCaptureFixture
) -> None:
    """An engine message may quote the request, so only its code is logged."""

    def echo(path: str, body: Any) -> tuple[int, dict[str, Any]]:
        if path == "/health":
            return 200, {"status": "ok", "running": 0, "queued": 0}
        return 422, {"error": f"invalid state: {body['state']}"}

    engine(echo)
    caplog.set_level(logging.DEBUG, logger="basal-worker")

    result = worker.handler({"input": {"requests": [one_question()]}})

    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert result["answers"][0]["error"]["code"] == "invalid_request"
    # The caller sees the engine's reason; the log does not (it may quote state).
    assert SENTINEL in result["answers"][0]["error"]["message"]
    assert SENTINEL not in logged
    assert "invalid_request" in logged


# --- the image is generic (EARS 1) -------------------------------------------

#: The only hosts the build context may name. An endpoint URL, a webhook or an
#: app link would make the image specific to one deployment — the whole point of
#: a generic worker is that it carries none of that. Loopback is the engine's
#: documented boundary; the rest are the engine's own source and licence texts.
ALLOWED_URL_HOSTS = (
    "github.com",
    "docs.runpod.io",
    "www.apache.org",
    "huggingface.co",
    "127.0.0.1",
    "localhost",
    # The container's view of the host, used only by the README's local
    # `/runsync` recipe (engine on the host, handler in the container).
    "host.docker.internal",
)

SECRET_PATTERNS = {
    "runpod api key": re.compile(r"rp_[A-Za-z0-9]{14,}"),
    "runpod endpoint": re.compile(r"api\.runpod\.ai/v2/"),
    "openai-style key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "bearer token": re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
    "key assignment": re.compile(r"(API_KEY|APIKEY|TOKEN|SECRET|PASSWORD)\s*[=:]\s*\S{8,}"),
}


def build_context_files() -> list[Path]:
    """Files that ship with the worker: the image contents plus its documents.

    `tests/` is excluded on purpose — its fixtures are loopback engines and
    fake payloads, and gitleaks (the CI job) scans the whole tree anyway.
    """
    skipped = {".git", "__pycache__", ".pytest_cache", ".venv", "tests"}
    return [
        path
        for path in sorted(WORKER.rglob("*"))
        if path.is_file() and not skipped & set(path.parts)
    ]


def _host_allowed(host: str) -> bool:
    return any(host == allowed or host.endswith(f".{allowed}") for allowed in ALLOWED_URL_HOSTS)


def test_build_context_has_no_secret_or_endpoint() -> None:
    """EARS 1: no key-shaped string, no endpoint id, no foreign URL."""
    offenders: list[str] = []
    for path in build_context_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                offenders.append(f"{path.name}: {name}")
        for url in re.findall(r"https?://[^\s\"')>]+", text):
            host = urlsplit(url).hostname or ""
            if not _host_allowed(host):
                offenders.append(f"{path.name}: url {url}")
    assert offenders == []


def test_dockerfile_pins_the_engine_and_the_defaults() -> None:
    """EARS 1: basal 2.3.1, text-only by default, media behind a flag."""
    dockerfile = (WORKER / "Dockerfile").read_text(encoding="utf-8")

    assert "ARG BASAL_VERSION=2.3.1" in dockerfile
    assert "refs/tags/v${BASAL_VERSION}.tar.gz" in dockerfile
    assert "2.2.0" not in dockerfile
    assert "BASAL_MODE=fast-nocompile" in dockerfile
    assert "BASAL_MEDIA=off" in dockerfile
    assert "BASAL_URL=http://127.0.0.1:8001" in dockerfile
    # The RunPod SDK owns 8000 (`--rp_serve_api`, /runsync); the engine must not.
    assert "BASAL_PORT=8001" in dockerfile
    assert "BASAL_PORT=8000" not in dockerfile


def test_no_calibration_or_weights_in_the_context() -> None:
    """Out of scope by contract: the calibration file arrives at deploy time."""
    names = {path.name for path in build_context_files()}
    assert not any(name.startswith("CALIBRATION") for name in names)
    assert not any(
        path.suffix in {".safetensors", ".bin", ".gguf"} for path in build_context_files()
    )


def test_worker_imports_only_the_stdlib() -> None:
    """`runpod` is imported inside `main()`: the image has it, this env does not.

    A third-party import at module level would make the worker depend on the
    product's package set, and the CI job that runs this suite in a throwaway
    environment (`uv run --no-project`) would need a `uv sync` to pass.
    """
    allowed = set(sys.stdlib_module_names) | {"runpod", "__future__"}
    offenders: list[str] = []
    for name in ("handler.py", "healthcheck.py"):
        tree = ast.parse((WORKER / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module.split(".")[0]]
            else:
                continue
            offenders += [f"{name}: {module}" for module in modules if module not in allowed]
    assert offenders == []
