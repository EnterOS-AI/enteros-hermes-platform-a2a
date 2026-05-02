"""End-to-end validation of MoleculeA2APlatformAdapter against a
hermes-agent install that has the ``register_platform_adapter`` patch.

Unlike ``tests/test_adapter.py`` (which unit-tests the adapter in
isolation), this script exercises the full production discovery path:

    pip install hermes-platform-molecule-a2a
        → hermes plugin manager scans entry_points group
            ``hermes_agent.plugins``
        → calls ``register(ctx)`` on the package
        → ``ctx.register_platform_adapter("molecule-a2a", ...)`` lands
            in the platform registry
        → ``GatewayConfig.from_dict({"platforms": {"molecule-a2a": ...}})``
            routes the entry into ``plugin_platforms``
        → ``GatewayRunner._create_plugin_adapter`` instantiates the
            class
        → adapter boots a real HTTP listener on an ephemeral port
        → POST /a2a/inbound dispatches a MessageEvent(internal=True)
            into the registered handler
        → ``adapter.send(...)`` POSTs to the per-chat callback URL
            learned from the inbound payload

Prerequisites:
    1. A hermes-agent checkout containing the ``register_platform_adapter``
       method on PluginContext + the GatewayConfig/GatewayRunner wiring.
       Set ``HERMES_REPO`` env to override the default (``~/.hermes/hermes-agent``).
    2. ``pip install -e .`` of this package into the same venv that
       runs hermes (so the entry point is registered).

Exit 0 on success.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List

HERMES_REPO = Path(
    os.environ.get("HERMES_REPO", str(Path.home() / ".hermes" / "hermes-agent"))
)
if not HERMES_REPO.exists():
    print(f"FAIL: hermes-agent checkout not found at {HERMES_REPO}")
    print("Set HERMES_REPO env to point at a fork with the "
          "register_platform_adapter patch.")
    sys.exit(1)
sys.path.insert(0, str(HERMES_REPO))

# Use a clean HERMES_HOME so user-dir plugins don't pollute the test —
# we want to confirm discovery via the pip entry_points path alone.
ROOT = Path(__file__).resolve().parent
os.environ["HERMES_HOME"] = str(ROOT / ".hermes_e2e_home")
(Path(os.environ["HERMES_HOME"]) / "plugins").mkdir(parents=True, exist_ok=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _amain() -> int:
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from hermes_cli.plugins import (
        discover_plugins,
        get_plugin_manager,
        get_plugin_platform_adapter,
    )

    # --- 1. Discovery via pip entry_points ----------------------------
    discover_plugins()
    mgr = get_plugin_manager()
    loaded = list(mgr._plugins.keys())
    assert "molecule_a2a" in loaded or any(
        "molecule" in n.lower() for n in loaded
    ), f"plugin not discovered via entry_points. Loaded: {loaded}. "\
       f"Run: pip install -e /Users/hongming/hermes-platform-molecule-a2a"
    print(f"OK: plugin discovered via pip entry_points ({len(loaded)} total)")

    # --- 2. Registry round-trip ---------------------------------------
    entry = get_plugin_platform_adapter("molecule-a2a")
    assert entry is not None, "molecule-a2a not in plugin platform registry"
    adapter_class, req_check = entry
    assert adapter_class.__name__ == "MoleculeA2APlatformAdapter", (
        f"unexpected class registered: {adapter_class.__name__}"
    )
    assert callable(req_check) and req_check() is True
    print(f"OK: registry returns {adapter_class.__name__} + requirements_check")

    # --- 3. GatewayConfig.from_dict routing ---------------------------
    listen_port = _free_port()
    cb_port = _free_port()
    cb_url = f"http://127.0.0.1:{cb_port}/reply"
    cfg_dict = {
        "platforms": {
            "molecule-a2a": {
                "enabled": True,
                "extra": {
                    "host": "127.0.0.1",
                    "port": listen_port,
                    "callback_url": cb_url,
                },
            },
        },
    }
    gc = GatewayConfig.from_dict(cfg_dict)
    assert "molecule-a2a" in gc.plugin_platforms, (
        f"GatewayConfig.from_dict didn't route molecule-a2a to "
        f"plugin_platforms; got {list(gc.plugin_platforms)}"
    )
    print("OK: GatewayConfig.from_dict routes molecule-a2a to plugin_platforms")

    # --- 4. _create_plugin_adapter instantiation ----------------------
    create_method = GatewayRunner._create_plugin_adapter

    class _Stub:
        pass

    stub = _Stub()
    pc = gc.plugin_platforms["molecule-a2a"]
    adapter = create_method(stub, "molecule-a2a", pc)
    assert adapter is not None, "_create_plugin_adapter returned None"
    assert adapter.__class__.__name__ == "MoleculeA2APlatformAdapter"
    print(f"OK: _create_plugin_adapter returned {adapter.__class__.__name__}")

    # --- 5. Boot the adapter on a real port ---------------------------
    received: List[MessageEvent] = []
    posted_replies: List[Dict[str, Any]] = []

    async def fake_handler(event: MessageEvent):
        received.append(event)
        return None  # don't auto-trigger send

    adapter.set_message_handler(fake_handler)

    # Stand up a fake molecule-runtime callback receiver.
    from aiohttp import web, ClientSession, ClientTimeout

    async def cb_handler(request: web.Request) -> web.Response:
        posted_replies.append(await request.json())
        return web.json_response({"ok": True})

    cb_app = web.Application()
    cb_app.router.add_post("/reply", cb_handler)
    cb_runner = web.AppRunner(cb_app)
    await cb_runner.setup()
    cb_site = web.TCPSite(cb_runner, "127.0.0.1", cb_port)
    await cb_site.start()

    try:
        ok = await adapter.connect()
        assert ok, "adapter.connect() returned False"
        print(f"OK: adapter listening on http://127.0.0.1:{listen_port}/a2a/inbound")

        # --- 6. Inbound A2A POST round-trip --------------------------
        async with ClientSession(timeout=ClientTimeout(total=5)) as s:
            url = f"http://127.0.0.1:{listen_port}/a2a/inbound"
            payload = {
                "chat_id": "peer-A",
                "peer_id": "peer-A",
                "peer_name": "ops-agent",
                "content": "ping from peer",
                "message_id": "m-1",
                "callback_url": cb_url,
            }
            async with s.post(url, json=payload) as r:
                assert r.status == 200, f"POST returned {r.status}"
                body = await r.json()
                assert body == {"ok": True, "queued": True}

        # handle_message dispatches in background; wait briefly.
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.02)
        assert len(received) == 1, "inbound message did not reach handler"
        ev = received[0]
        assert ev.text == "ping from peer"
        assert ev.internal is True
        assert ev.source.chat_id == "peer-A"
        print("OK: inbound POST → MessageEvent(internal=True) → message_handler")

        # --- 7. Outbound send() round-trip ---------------------------
        result = await adapter.send(
            chat_id="peer-A",
            content="pong from agent",
            reply_to="m-1",
            metadata={"source": "test"},
        )
        assert result.success, f"send() failed: {result.error}"
        assert len(posted_replies) == 1
        assert posted_replies[0] == {
            "chat_id": "peer-A",
            "content": "pong from agent",
            "reply_to": "m-1",
            "metadata": {"source": "test"},
        }
        print("OK: send() POSTed reply to per-chat callback URL")

    finally:
        await adapter.disconnect()
        await cb_site.stop()
        await cb_runner.cleanup()

    # --- 8. Session round-trip survives daemon restart ----------------
    # Plugin platforms die without resolve_platform_id: SessionSource.
    # from_dict calls Platform(data["platform"]) which would raise
    # ValueError for "molecule-a2a". Confirm the patched fork accepts it.
    from gateway.session import SessionSource
    from hermes_cli.plugins import PluginPlatformIdentifier

    src = ev.source
    serialized = src.to_dict()
    assert serialized["platform"] == "molecule-a2a"
    restored = SessionSource.from_dict(serialized)
    assert isinstance(restored.platform, PluginPlatformIdentifier), (
        f"SessionSource.from_dict returned {type(restored.platform).__name__} "
        "for plugin platform — resolve_platform_id fallback not in place"
    )
    assert restored.platform.value == "molecule-a2a"
    assert restored.chat_id == "peer-A"
    print("OK: SessionSource survives to_dict/from_dict round-trip")

    print("\n✓ Full E2E pipeline validated:")
    print("  pip entry_points → plugin discovery → platform registry")
    print("  → GatewayConfig.from_dict → _create_plugin_adapter")
    print("  → live HTTP listener → MessageEvent dispatch → callback POST")
    print("  → SessionSource serialization round-trip (plugin-platform-safe)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_amain()))
