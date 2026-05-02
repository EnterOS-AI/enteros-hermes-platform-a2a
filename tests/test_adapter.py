"""Unit + integration tests for MoleculeA2APlatformAdapter.

What we cover:
  - Plugin entry point: register() claims the platform name through
    PluginContext.register_platform_adapter
  - Adapter init: subclasses BasePlatformAdapter cleanly, platform
    identifier has the right .value
  - HTTP listener: connect/disconnect lifecycle on a real ephemeral
    port; health endpoint reachable
  - Inbound: POST /a2a/inbound with valid payload reaches
    self.handle_message → invokes the registered message handler
  - Inbound auth: shared_secret enforced; missing/bad header returns
    401; missing fields return 400
  - Outbound: send() POSTs to the per-chat callback URL learned from
    the inbound payload, falls back to default_callback_url, errors
    when neither is set
"""

from __future__ import annotations

import asyncio
import socket
from typing import Any, Dict, List

import pytest
from aiohttp import ClientSession, ClientTimeout, web

from hermes_platform_molecule_a2a import (
    MoleculeA2APlatformAdapter,
    check_molecule_a2a_requirements,
    register,
)
from hermes_platform_molecule_a2a.adapter import (
    HEALTH_PATH,
    INBOUND_PATH,
    PLATFORM_NAME,
    SECRET_HEADER,
)

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent
from hermes_cli.plugins import PluginPlatformIdentifier


# ---- helpers --------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_adapter(**extra: Any) -> MoleculeA2APlatformAdapter:
    cfg = PlatformConfig(enabled=True, extra=extra)
    return MoleculeA2APlatformAdapter(cfg)


# ---- structural tests ----------------------------------------------


def test_check_requirements_returns_true_when_aiohttp_present():
    assert check_molecule_a2a_requirements() is True


def test_adapter_subclasses_base_and_uses_plugin_identifier():
    adapter = _make_adapter()
    assert isinstance(adapter, BasePlatformAdapter)
    assert isinstance(adapter.platform, PluginPlatformIdentifier)
    assert adapter.platform.value == PLATFORM_NAME


def test_register_calls_plugin_context_with_correct_name():
    """The entry-point register() must claim the platform name and pass
    the adapter class + requirements check. We capture the args by
    feeding it a fake PluginContext."""
    captured: Dict[str, Any] = {}

    class FakeCtx:
        def register_platform_adapter(self, *, name, adapter_class, requirements_check=None):
            captured["name"] = name
            captured["adapter_class"] = adapter_class
            captured["requirements_check"] = requirements_check

    register(FakeCtx())

    assert captured["name"] == PLATFORM_NAME
    assert captured["adapter_class"] is MoleculeA2APlatformAdapter
    assert captured["requirements_check"] is check_molecule_a2a_requirements
    assert captured["requirements_check"]() is True


# ---- live HTTP tests -----------------------------------------------


@pytest.mark.asyncio
async def test_connect_starts_listener_and_health_responds():
    port = _free_port()
    adapter = _make_adapter(host="127.0.0.1", port=port)
    try:
        ok = await adapter.connect()
        assert ok is True
        assert adapter.is_connected is True

        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.get(f"http://127.0.0.1:{port}{HEALTH_PATH}") as r:
                assert r.status == 200
                body = await r.json()
                assert body == {"ok": True, "platform": PLATFORM_NAME}
    finally:
        await adapter.disconnect()
    assert adapter.is_connected is False


@pytest.mark.asyncio
async def test_inbound_post_dispatches_to_message_handler():
    port = _free_port()
    adapter = _make_adapter(host="127.0.0.1", port=port)

    received: List[MessageEvent] = []

    async def handler(event: MessageEvent):
        received.append(event)
        return None  # don't trigger a reply

    adapter.set_message_handler(handler)

    try:
        await adapter.connect()
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            payload = {
                "chat_id": "peer-uuid-1",
                "peer_id": "peer-uuid-1",
                "peer_name": "ops-agent",
                "content": "hello from peer",
                "message_id": "msg-1",
                "callback_url": "http://example.test/reply",
            }
            async with session.post(
                f"http://127.0.0.1:{port}{INBOUND_PATH}", json=payload
            ) as r:
                assert r.status == 200
                body = await r.json()
                assert body == {"ok": True, "queued": True}

        # handle_message dispatches in a background task.
        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.02)
    finally:
        await adapter.disconnect()

    assert len(received) == 1
    event = received[0]
    assert event.text == "hello from peer"
    assert event.internal is True
    assert event.source.chat_id == "peer-uuid-1"
    assert event.source.user_name == "ops-agent"
    # Per-chat callback was learned from the payload.
    assert adapter._callbacks["peer-uuid-1"] == "http://example.test/reply"


@pytest.mark.asyncio
async def test_inbound_rejects_when_shared_secret_mismatches():
    port = _free_port()
    adapter = _make_adapter(host="127.0.0.1", port=port, shared_secret="topsecret")
    received: List[MessageEvent] = []
    adapter.set_message_handler(
        lambda event: received.append(event) or None  # type: ignore[func-returns-value]
    )

    try:
        await adapter.connect()
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            url = f"http://127.0.0.1:{port}{INBOUND_PATH}"
            payload = {"chat_id": "x", "content": "hi"}

            # No header → 401.
            async with session.post(url, json=payload) as r:
                assert r.status == 401

            # Wrong secret → 401.
            async with session.post(
                url, json=payload, headers={SECRET_HEADER: "wrong"}
            ) as r:
                assert r.status == 401

            # Correct secret → 200, dispatched.
            async with session.post(
                url, json=payload, headers={SECRET_HEADER: "topsecret"}
            ) as r:
                assert r.status == 200

        for _ in range(50):
            if received:
                break
            await asyncio.sleep(0.02)
    finally:
        await adapter.disconnect()

    assert len(received) == 1


@pytest.mark.asyncio
async def test_inbound_rejects_missing_required_fields():
    port = _free_port()
    adapter = _make_adapter(host="127.0.0.1", port=port)
    try:
        await adapter.connect()
        url = f"http://127.0.0.1:{port}{INBOUND_PATH}"
        async with ClientSession(timeout=ClientTimeout(total=5)) as session:
            async with session.post(url, json={"content": "hi"}) as r:
                assert r.status == 400
            async with session.post(url, json={"chat_id": "x"}) as r:
                assert r.status == 400
            async with session.post(url, data="not json") as r:
                assert r.status == 400
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_send_posts_to_per_chat_callback_url():
    """send() must POST to the callback URL learned from the most
    recent inbound message for that chat_id."""

    received_posts: List[Dict[str, Any]] = []

    async def callback_handler(request: web.Request) -> web.Response:
        received_posts.append({
            "headers": dict(request.headers),
            "body": await request.json(),
        })
        return web.json_response({"ok": True})

    callback_app = web.Application()
    callback_app.router.add_post("/reply", callback_handler)
    callback_runner = web.AppRunner(callback_app)
    await callback_runner.setup()
    cb_port = _free_port()
    callback_site = web.TCPSite(callback_runner, "127.0.0.1", cb_port)
    await callback_site.start()

    callback_url = f"http://127.0.0.1:{cb_port}/reply"
    adapter = _make_adapter(host="127.0.0.1", port=_free_port())
    adapter._callbacks["chat-A"] = callback_url

    try:
        result = await adapter.send(
            chat_id="chat-A",
            content="hello back",
            reply_to="msg-1",
            metadata={"thread_id": "t"},
        )
        assert result.success is True
        assert len(received_posts) == 1
        assert received_posts[0]["body"] == {
            "chat_id": "chat-A",
            "content": "hello back",
            "reply_to": "msg-1",
            "metadata": {"thread_id": "t"},
        }
    finally:
        await callback_site.stop()
        await callback_runner.cleanup()


@pytest.mark.asyncio
async def test_send_falls_back_to_default_callback_url():
    received_posts: List[Dict[str, Any]] = []

    async def callback_handler(request: web.Request) -> web.Response:
        received_posts.append(await request.json())
        return web.json_response({"ok": True})

    callback_app = web.Application()
    callback_app.router.add_post("/default-reply", callback_handler)
    callback_runner = web.AppRunner(callback_app)
    await callback_runner.setup()
    cb_port = _free_port()
    callback_site = web.TCPSite(callback_runner, "127.0.0.1", cb_port)
    await callback_site.start()

    default_cb = f"http://127.0.0.1:{cb_port}/default-reply"
    adapter = _make_adapter(host="127.0.0.1", port=_free_port(), callback_url=default_cb)
    # No per-chat entry — must fall back to default.
    try:
        result = await adapter.send(chat_id="never-seen", content="hi")
        assert result.success is True
        assert received_posts[0]["chat_id"] == "never-seen"
    finally:
        await callback_site.stop()
        await callback_runner.cleanup()


@pytest.mark.asyncio
async def test_send_returns_error_when_no_callback_configured():
    adapter = _make_adapter(host="127.0.0.1", port=_free_port())
    result = await adapter.send(chat_id="anything", content="hi")
    assert result.success is False
    assert result.retryable is False
    assert "callback_url" in (result.error or "")


@pytest.mark.asyncio
async def test_get_chat_info_returns_minimal_metadata():
    adapter = _make_adapter()
    info = await adapter.get_chat_info("chat-X")
    assert info == {"name": "chat-X", "type": "dm", "chat_id": "chat-X"}
