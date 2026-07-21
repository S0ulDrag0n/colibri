#!/usr/bin/env python3
"""
stdlib-only healthcheck for the colibrì container.

Pings the OpenAI-compatible API exposed by `coli serve`. This is a
**container-alive** check, NOT a model-readiness check:

  - The engine takes 30+ seconds to load the 370 GB int4 model and an
    additional 10-30 seconds to warm up the expert cache. A real readiness
    probe would have to wait for that, which would falsely fail any CI
    smoke test that runs every minute.
  - During the load window the engine is busy and not yet accepting HTTP
    requests, so we expect transient connection errors — exit code 1 from
    the healthcheck during the start_period is the correct behaviour.
  - The check is a single GET to /v1/models, which the engine serves
    immediately after it has bound the listening socket. The wait for
    `/v1/chat/completions` to be usable is implicit: once /v1/models
    returns 200, the OpenAI gateway thread is already up.

Exits 0 on any 2xx response, 1 on any other outcome (DNS, connection
refused, timeout, non-2xx, etc.). The compose healthcheck uses retries +
start_period to absorb the load window — see docker-compose.yml.

We deliberately avoid third-party libraries (no requests, no urllib3)
because the runtime container has no pip dependencies; this script must
work in a clean python3 install.
"""
import os
import sys
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8000/v1/models"
TIMEOUT_SECONDS = 3


def main() -> int:
    url = os.environ.get("COLI_HEALTHCHECK_URL", DEFAULT_URL)
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp:
            return 0 if 200 <= resp.status < 300 else 1
    except urllib.error.HTTPError as e:
        # Server replied but with a 4xx/5xx — the gateway is up, just unhappy.
        # Treat 4xx as "alive" (the URL exists), 5xx as "not yet" (1).
        return 0 if 400 <= e.code < 500 else 1
    except (urllib.error.URLError, OSError, TimeoutError):
        # Connection refused, DNS failure, timeout — engine is still loading
        # or the container is in a bad state. Either way, the check fails.
        return 1


if __name__ == "__main__":
    sys.exit(main())
