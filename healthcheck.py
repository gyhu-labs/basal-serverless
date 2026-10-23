"""Docker ``HEALTHCHECK``: the engine answers ``/health`` with ``status: ok``.

basal 2.3.x reports ``{"status": "ok", "running": R, "queued": Q}``; the counts
are passed through in the message so `docker inspect` shows whether a probe
landed while the engine was busy rather than idle. This is the container's own
liveness signal — the batch gate the gateway sees is the handler's (180 s, then
`worker_unhealthy`), not this one.
"""

import json
import os
import sys
import urllib.request

URL = os.environ.get("BASAL_URL", "http://127.0.0.1:8001").rstrip("/") + "/health"

try:
    with urllib.request.urlopen(URL, timeout=4) as response:
        body = json.load(response)
except (OSError, ValueError) as exc:
    print(f"engine health unreadable: {exc.__class__.__name__}")
    sys.exit(1)

if not isinstance(body, dict):
    print("engine health is not a JSON object")
    sys.exit(1)

print(f"status={body.get('status')} running={body.get('running')} queued={body.get('queued')}")
sys.exit(0 if body.get("status") == "ok" else 1)
