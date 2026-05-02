"""End-to-end validation against a real `hermes gateway run` subprocess.

Where ``e2e_validate.py`` exercises the plugin in-process (importing
``hermes_cli.plugins`` directly), this script spawns the actual hermes
binary as the daemon does in production. It validates:

  - Plugin discovery via pip ``entry_points`` survives the real
    `PluginManager` boot sequence inside a fresh interpreter.
  - The seeded ``platforms.molecule-a2a`` config block in
    ``~/.hermes/config.yaml`` is parsed and routed to ``plugin_platforms``
    by ``GatewayConfig.from_dict`` inside the real boot path.
  - ``GatewayRunner._run_async`` instantiates the plugin adapter and
    ``connect()`` actually opens the HTTP listener on the configured port.
  - The /a2a/health endpoint is reachable from outside the subprocess.
  - A real /a2a/inbound POST gets a 200 ack back from the running gateway.

This is the closest reproduction of production-shape boot we can run
without provisioning a workspace + having an LLM provider key + having
a peer agent. The 8/8 in-process E2E plus this script form the gating
evidence for the upstream PR's "validated end-to-end" claim.

Pre-reqs:
    1. Patched hermes fork installed in a venv with the plugin pip-
       installed (the plugin's tests/ + scripts/ assume the venv at
       ~/.hermes/hermes-agent/venv).
    2. A working `hermes` binary on PATH or at the venv default path
       (override with HERMES_BIN env).

Run:
    python scripts/e2e_real_hermes_subprocess.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HERMES_BIN = str(
    Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "hermes"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_url(url: str, timeout_secs: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.25)
        except Exception:
            time.sleep(0.25)
    return False


def _post_json(url: str, payload: dict) -> tuple[int, str]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def main() -> int:
    hermes_bin = os.environ.get("HERMES_BIN", DEFAULT_HERMES_BIN)
    if not Path(hermes_bin).exists():
        print(f"FAIL: hermes binary not found at {hermes_bin}")
        print("Set HERMES_BIN env to override.")
        return 1

    listen_port = _free_port()
    cb_port = _free_port()
    cb_url = f"http://127.0.0.1:{cb_port}/reply"

    with tempfile.TemporaryDirectory(prefix="hermes-e2e-") as tmp:
        tmp = Path(tmp)
        hermes_home = tmp / ".hermes"
        hermes_home.mkdir()

        # Minimal config — model.provider just has to parse; no actual
        # LLM call is made by this test. The plugin platform stanza is
        # the actual unit under test.
        (hermes_home / "config.yaml").write_text(
            "model:\n"
            "  default: \"nousresearch/hermes-4-70b\"\n"
            "  provider: \"openrouter\"\n"
            "platforms:\n"
            "  molecule-a2a:\n"
            "    enabled: true\n"
            "    extra:\n"
            "      host: \"127.0.0.1\"\n"
            f"      port: {listen_port}\n"
            f"      callback_url: \"{cb_url}\"\n"
        )
        (hermes_home / ".env").write_text(
            "OPENROUTER_API_KEY=sk-stub-not-used\n"
        )
        print(f"OK: wrote tmp HERMES_HOME at {hermes_home}")

        env = {
            **os.environ,
            "HOME": str(tmp),
            "HERMES_HOME": str(hermes_home),
        }

        log_file = open(tmp / "gateway.log", "w+", buffering=1)
        proc = subprocess.Popen(
            [hermes_bin, "gateway", "run"],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(tmp),
        )
        print(f"OK: spawned `{hermes_bin} gateway run` as pid {proc.pid}")

        try:
            health_url = f"http://127.0.0.1:{listen_port}/a2a/health"
            if not _wait_url(health_url, timeout_secs=45.0):
                proc.terminate()
                proc.wait(timeout=5)
                print(f"FAIL: /a2a/health not reachable within 45s")
                print("--- gateway log tail:")
                log_file.seek(0)
                print(log_file.read()[-4000:])
                return 1
            print(f"OK: GET {health_url} responded 200")

            inbound_url = f"http://127.0.0.1:{listen_port}/a2a/inbound"
            status, body = _post_json(inbound_url, {
                "chat_id": "peer-real-1",
                "peer_id": "peer-real-1",
                "peer_name": "ops-agent",
                "content": "hello from a real subprocess test",
                "message_id": "msg-real-1",
                "callback_url": cb_url,
            })
            assert status == 200, f"inbound POST returned {status}: {body}"
            assert json.loads(body) == {"ok": True, "queued": True}
            print(f"OK: POST {inbound_url} returned 200 with queued ack")

            # The 200s above already prove the plugin booted — both
            # /a2a/health and /a2a/inbound are served by the plugin's
            # own aiohttp app. If the adapter's connect() didn't run,
            # neither endpoint would respond.

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log_file.close()

    print("\n✓ Real-subprocess E2E passed:")
    print("  fresh-interpreter plugin discovery → real GatewayRunner boot")
    print("  → real HTTP listener → real /a2a/inbound roundtrip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
