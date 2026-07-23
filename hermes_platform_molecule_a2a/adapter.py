"""Molecule A2A platform adapter for Hermes.

Architecture
============
Hermes runs as a long-lived gateway daemon inside a Molecule workspace
container. The molecule-runtime that wraps the workspace owns the A2A
inbox: peer agents POST A2A messages to it, it queues them, and a
"runtime" component decides what to do with each one.

For every other LLM runtime (claude-code MCP push, codex app-server)
the runtime hands the message to the LLM via a native push mechanism so
the LLM keeps a single coherent session across messages. Hermes is the
last runtime missing that — until this plugin lands, every A2A message
spawns a fresh `hermes` subprocess against stateless `/v1/chat/completions`,
so peer-agent conversations have no continuity.

This adapter closes that gap by giving molecule-runtime a stable HTTP
target on the running hermes daemon. The flow is symmetric:

    inbound  : runtime → POST /a2a/inbound → MessageEvent(internal=True)
                                          → handle_message → agent reply
    outbound : send(chat_id, content)      → POST <callback_url>
                                          → runtime delivers to peer

`internal=True` on the MessageEvent bypasses the per-platform user
allowlist check at gateway/run.py — A2A messages are pre-authorized by
the platform layer (peer registry + tenant isolation) before they ever
reach hermes.

Configuration
=============
Loaded from ``platforms.molecule-a2a`` in hermes config.yaml::

    platforms:
      molecule-a2a:
        enabled: true
        extra:
          host: "127.0.0.1"          # default; bind localhost only
          port: 8645                  # default
          callback_url: "..."         # default outbound target if the
                                      # inbound message didn't carry one
          shared_secret: "..."        # required unless empty; checked
                                      # against X-Molecule-A2A-Secret

Inbound payload shape::

    {
      "chat_id":      "<peer_id or session_key>",
      "peer_id":      "<peer agent UUID>",
      "peer_name":    "ops-agent",
      "peer_role":    "sre",
      "content":      "the message text",
      "message_id":   "uuid-or-monotonic",
      "callback_url": "http://runtime:9999/a2a/reply",  # optional
      "thread_id":    "optional"
    }
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from typing import Any, Dict, Optional

try:
    from aiohttp import ClientSession, ClientTimeout, web
    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - presence checked by requirements
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]
    ClientSession = None  # type: ignore[assignment]
    ClientTimeout = None  # type: ignore[assignment]

from gateway.config import PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.config import Platform

# Import delegation_tools lazily so the adapter loads without httpx
# being a hard import-time requirement (hermes already depends on aiohttp;
# httpx is added as a dep of this plugin). If the import fails hermes
# will still boot but delegation tools won't work.
_deLEGATION_TOOLS: Any = None


def _get_delegation_tools():
    global _deLEGATION_TOOLS
    if _deLEGATION_TOOLS is None:
        try:
            from . import delegation_tools as dt  # type: ignore[attr-defined]
            _deLEGATION_TOOLS = dt
        except Exception:  # pragma: no cover - defensive, module already tested
            _deLEGATION_TOOLS = None
    return _deLEGATION_TOOLS


def _platform_identity(name: str):
    """Pick the right Platform-shaped identity for the installed hermes.

    Upstream NousResearch/hermes-agent#17751 (merged 2026-04-30) made
    Platform an open enum (``Platform("molecule-a2a")`` works via
    ``_missing_()``). Pre-#17751 forks have a closed enum + ship
    ``PluginPlatformIdentifier`` for plugin-supplied platforms instead.
    Detect at import time so the same plugin works on both.
    """
    try:
        return Platform(name)
    except ValueError:
        from hermes_cli.plugins import PluginPlatformIdentifier
        return PluginPlatformIdentifier(name)


logger = logging.getLogger(__name__)

PLATFORM_NAME = "molecule-a2a"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8645
SECRET_HEADER = "X-Molecule-A2A-Secret"
INBOUND_PATH = "/a2a/inbound"
HEALTH_PATH = "/a2a/health"


def check_molecule_a2a_requirements() -> bool:
    """Hermes calls this before instantiating the adapter."""
    return AIOHTTP_AVAILABLE


class MoleculeA2APlatformAdapter(BasePlatformAdapter):
    """Receive A2A peer messages over HTTP, dispatch into the hermes
    gateway, and POST agent replies back to the molecule-runtime."""

    # hermes >= 0.19 authz contract (gateway/authz_mixin.py): declaring
    # authorization_is_upstream makes the gateway honor THIS adapter's
    # intake decision instead of applying its pairing/allowlist DM
    # policies. That is correct here: every inbound message is
    # authenticated upstream by the platform shared-secret
    # (X-Molecule-A2A-Secret, hmac.compare_digest in _handle_inbound)
    # on a localhost-only listener — there are no "unknown senders" to
    # pair. Without this flag, 0.19's default pairing policy silently
    # DROPPED platform messages mid-session ("Dropping message from
    # unauthorized user in active session", 2026-07-23) and the
    # executor's pending future expired as a 600s timeout bubble.
    authorization_is_upstream = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, _platform_identity(PLATFORM_NAME))
        extra = getattr(config, "extra", None) or {}
        self._host: str = extra.get("host", DEFAULT_HOST)
        self._port: int = int(extra.get("port", DEFAULT_PORT))
        self._default_callback_url: Optional[str] = extra.get("callback_url")
        self._shared_secret: str = str(extra.get("shared_secret", "") or "")
        # Optional platform URL and workspace ID for delegation tools.
        # In-container hermes: PLATFORM_URL defaults to host.docker.internal:8080.
        # External hermes (via hermes-channel-molecule): set these in the
        # config so the delegation tools can reach the platform.
        self._platform_url: str = extra.get(
            "platform_url", "http://host.docker.internal:8080"
        )
        self._workspace_id: str = extra.get("workspace_id", "")
        # Push config into delegation_tools so hermes can call list_peers etc.
        # without the module having to re-import the adapter or hold circular refs.
        dt = _get_delegation_tools()
        if dt is not None:
            dt.configure(
                platform_url=self._platform_url,
                workspace_id=self._workspace_id,
            )
        self._runner: Optional[Any] = None  # aiohttp AppRunner
        self._site: Optional[Any] = None    # aiohttp TCPSite
        # Per-chat callback URL learned from the inbound payload. Lets
        # send() POST replies back to whichever runtime endpoint
        # delivered the original message.
        self._callbacks: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "Molecule-A2A"

    # ---- lifecycle ----------------------------------------------------

    # ``is_reconnect`` (hermes >= 0.19): the gateway's base-adapter ABC
    # grew a keyword-only flag (False on cold boot, True when the
    # reconnect watcher re-establishes a dropped platform). Accept and
    # ignore it — our listener setup is identical either way — but keep
    # the default so pre-0.19 gateways (which call connect() bare)
    # still work. Without this, 0.19 boot fails with "unexpected
    # keyword argument 'is_reconnect'" and :8645 never binds
    # (2026-07-23 first-boot hang).
    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp is required for molecule-a2a adapter")
            return False

        app = web.Application()
        app.router.add_post(INBOUND_PATH, self._handle_inbound)
        app.router.add_get(HEALTH_PATH, self._handle_health)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        self._mark_connected()
        logger.info(
            "molecule-a2a listening on http://%s:%d%s",
            self._host, self._port, INBOUND_PATH,
        )
        return True

    async def disconnect(self) -> None:
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:
                logger.exception("molecule-a2a: site stop failed")
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                logger.exception("molecule-a2a: runner cleanup failed")
            self._runner = None
        self._mark_disconnected()

    # ---- inbound (HTTP → MessageEvent) -------------------------------

    async def _handle_health(self, _request: "web.Request") -> "web.Response":
        return web.json_response({"ok": True, "platform": PLATFORM_NAME})

    async def _handle_inbound(self, request: "web.Request") -> "web.Response":
        # Constant-time secret comparison. Empty shared_secret = open
        # mode (intended for in-container localhost-only deployments
        # where the network layer is the trust boundary).
        if self._shared_secret:
            provided = request.headers.get(SECRET_HEADER, "")
            if not hmac.compare_digest(provided, self._shared_secret):
                return web.json_response(
                    {"ok": False, "error": "unauthorized"}, status=401
                )

        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"ok": False, "error": "invalid json"}, status=400
            )

        if not isinstance(payload, dict):
            return web.json_response(
                {"ok": False, "error": "expected object"}, status=400
            )

        chat_id = payload.get("chat_id") or payload.get("peer_id")
        content = payload.get("content")
        if not chat_id or not isinstance(content, str):
            return web.json_response(
                {"ok": False, "error": "chat_id and content required"},
                status=400,
            )

        callback_url = payload.get("callback_url") or self._default_callback_url
        if callback_url:
            self._callbacks[str(chat_id)] = str(callback_url)

        peer_name = payload.get("peer_name")
        source = self.build_source(
            chat_id=str(chat_id),
            chat_name=peer_name or str(chat_id),
            chat_type="dm",
            user_id=str(payload.get("peer_id") or chat_id),
            user_name=peer_name,
            thread_id=payload.get("thread_id"),
        )

        # Belt-and-suspenders session continuity (task #385): if the
        # caller shipped messages_history AND the daemon's session for
        # this chat is fresh (no transcript yet, or shorter than the
        # supplied history), seed the transcript so the agent has the
        # prior turns. No-op once the daemon's own SessionStore has
        # accumulated the same turns — i.e. on every subsequent turn
        # within the same container lifecycle. Skipped silently if
        # messages_history is malformed; the turn still dispatches.
        self._maybe_seed_transcript(source, payload.get("messages_history"))

        event = MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            message_id=str(payload.get("message_id") or ""),
            internal=True,
            raw_message=payload,
        )

        # handle_message returns quickly — it spawns a background task.
        # Ack the HTTP request immediately so the runtime is unblocked
        # to deliver the next message.
        asyncio.create_task(self.handle_message(event))
        return web.json_response({"ok": True, "queued": True})

    def _maybe_seed_transcript(
        self,
        source: Any,
        history: Any,
    ) -> None:
        """Seed daemon transcript from canvas-shipped history on a fresh
        session. See class docstring + task #385 for rationale."""
        if not isinstance(history, list) or not history:
            return
        store = getattr(self, "_session_store", None)
        if store is None:
            return
        try:
            entry = store.get_or_create_session(source)
            existing = store.load_transcript(entry.session_id)
            if len(existing) >= len(history):
                return  # daemon already has at least as much — don't re-seed
            for turn in history:
                if not isinstance(turn, dict):
                    continue
                store.append_to_transcript(entry.session_id, turn)
        except Exception:
            logger.exception("molecule-a2a: transcript seed failed; continuing")

    # ---- outbound (send → HTTP POST to callback) ---------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        callback_url = self._callbacks.get(chat_id) or self._default_callback_url
        if not callback_url:
            return SendResult(
                success=False,
                error=(
                    "no callback_url for chat_id and no default configured — "
                    "set platforms.molecule-a2a.extra.callback_url or include "
                    "callback_url in the inbound payload"
                ),
                retryable=False,
            )

        body = {
            "chat_id": chat_id,
            "content": content,
            "reply_to": reply_to,
            "metadata": metadata or {},
        }
        headers: Dict[str, str] = {}
        if self._shared_secret:
            headers[SECRET_HEADER] = self._shared_secret

        try:
            async with ClientSession(timeout=ClientTimeout(total=30)) as session:
                async with session.post(callback_url, json=body, headers=headers) as resp:
                    text = await resp.text()
                    if 200 <= resp.status < 300:
                        return SendResult(success=True, raw_response=text)
                    return SendResult(
                        success=False,
                        error=f"callback returned HTTP {resp.status}: {text[:200]}",
                        retryable=resp.status >= 500,
                    )
        except asyncio.TimeoutError:
            return SendResult(
                success=False,
                error="callback POST timed out after 30s",
                retryable=True,
            )
        except Exception as exc:
            return SendResult(
                success=False,
                error=f"callback POST failed: {exc}",
                retryable=True,
            )

    # ---- required by base contract -----------------------------------

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    # send_typing has a usable default in base (no-op).
