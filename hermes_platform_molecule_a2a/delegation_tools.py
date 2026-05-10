"""Molecule A2A delegation tools callable from the hermes agent.

Implements Option A of the #157 fix: expose ``list_peers`` and
``delegate_task`` as first-class tools inside hermes workspaces that
use the molecule-a2a platform adapter.

Why these tools live here rather than in the platform runtime:
  - The platform-side delegation endpoints (``POST /workspaces/:id/delegate``,
    ``GET /workspaces/:id/delegations``) are HTTP APIs available via the
    platform URL that every workspace container already knows (``PLATFORM_URL``).
  - We need no hermes-specific imports; ``httpx`` is the only runtime dep.
  - The adapter's ``__init__`` can inject ``PLATFORM_URL`` and ``WORKSPACE_ID``
    from the config so hermes workspaces running on any host can reach the
    platform without hard-coding URLs.

Architecture::

    hermes agent
        → ctx.register_tool()   [hermes calls this at boot]
        → tool_list_peers      [ctx.tool() call]
        → HTTP GET /registry/{workspace_id}/peers
        → list of peer dicts

    hermes agent
        → ctx.register_tool()
        → tool_delegate_task    [ctx.tool() call]
        → HTTP POST /workspaces/{workspace_id}/delegate
        → platform dispatches A2A message/send to target peer
        → poll GET /workspaces/{workspace_id}/delegations until terminal
        → return response_text

The tool signatures match the molecule-runtime MCP tool surface so that
the same prompt-injected "how to delegate" instructions work in hermes
as they do in the claude-code and codex runtimes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Module-level configuration — set once per adapter instance at boot.
# Thread-safe reads; written once at hermes startup before any tool calls.
_platform_url: str = "http://host.docker.internal:8080"
_workspace_id: str = ""
_platform_url_lock = threading.Lock()


def configure(platform_url: str | None = None, workspace_id: str | None = None) -> None:
    """Apply adapter-level config to the delegation tools module.

    Called by ``MoleculeA2APlatformAdapter.__init__`` so that delegation
    tool calls always hit the right platform URL even when hermes is
    running on a remote host (e.g. via hermes-channel-molecule relay).

    Idempotent — safe to call multiple times in dev/test scenarios.
    """
    global _platform_url, _workspace_id
    if platform_url is not None:
        with _platform_url_lock:
            _platform_url = platform_url
    if workspace_id is not None:
        _workspace_id = workspace_id


# Polling parameters for sync delegation.
_POLL_INTERVAL_S = 3.0
_POLL_BUDGET_S = float(os.environ.get("DELEGATION_TIMEOUT", "300.0"))
_POLL_MAX_ATTEMPTS = int(_POLL_BUDGET_S / _POLL_INTERVAL_S)

_A2A_ERROR_PREFIX = "ERROR: "
_DELEGATION_ERROR_PREFIX = "DELEGATION ERROR: "


def _auth_headers(source_id: str) -> dict[str, str]:
    """Build auth headers for platform calls from a workspace container.

    The platform validates the bearer token bound to the calling workspace.
    Inside a container, WORKSPACE_AUTH_TOKEN is set by the platform provisioner.
    """
    token = os.environ.get("WORKSPACE_AUTH_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "X-Workspace-ID": source_id or _workspace_id,
    }


# ---------------------------------------------------------------------------
# Tool: list_peers
# ---------------------------------------------------------------------------

TOOL_SCHEMA_LIST_PEERS = {
    "name": "list_peers",
    "description": (
        "List all peer workspaces visible to this workspace from the "
        "platform registry. Each peer dict contains: id, name, role, "
        "status, tier, url. Returns an empty list if the platform is "
        "unreachable or no peers are registered."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def tool_list_peers(source_workspace_id: str | None = None) -> list[dict[str, Any]]:
    """Get this workspace's peers from the platform registry.

    Args:
        source_workspace_id: Optional. Override which workspace to query
            as the source (for multi-workspace agents that act as a router).

    Returns:
        List of peer dicts from the registry. Empty list on any error.
    """
    src = source_workspace_id or _workspace_id
    headers = _auth_headers(src)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_platform_url}/registry/{src}/peers",
                headers=headers,
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(
                "list_peers: registry returned %s for workspace %s",
                resp.status_code,
                src,
            )
            return []
    except Exception as e:
        logger.warning("list_peers: failed to reach platform: %s", e)
        return []


# ---------------------------------------------------------------------------
# Tool: delegate_task
# ---------------------------------------------------------------------------

TOOL_SCHEMA_DELEGATE_TASK = {
    "name": "delegate_task",
    "description": (
        "Send a task to a peer workspace and wait for its response. "
        "Uses the platform's async delegation endpoint with polling for "
        "the result (bypasses the platform's 600s HTTP timeout ceiling). "
        "Returns the target workspace's response text, or an error string."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "The target workspace UUID to delegate to.",
            },
            "task": {
                "type": "string",
                "description": "The task description / message to send.",
            },
            "source_workspace_id": {
                "type": "string",
                "description": (
                    "Optional. Override the source workspace for the "
                    "delegation (for multi-workspace agents)."
                ),
            },
        },
        "required": ["workspace_id", "task"],
    },
}


async def tool_delegate_task(
    workspace_id: str,
    task: str,
    source_workspace_id: str | None = None,
) -> str:
    """Send a task to a peer workspace via the platform and return the response.

    Args:
        workspace_id: The target workspace UUID.
        task: The task text to send.
        source_workspace_id: Override the source workspace (multi-workspace mode).

    Returns:
        The target's response text, or an error prefixed with ``ERROR:``.
    """
    src = source_workspace_id or _workspace_id
    headers = _auth_headers(src)

    # Compute an idempotency key scoped to this delegation so retries
    # during a container restart don't duplicate the work.
    idem_key = hashlib.sha256(
        f"{src}:{workspace_id}:{task}".encode()
    ).hexdigest()[:32]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_platform_url}/workspaces/{src}/delegate",
                json={
                    "target_id": workspace_id,
                    "task": task,
                    "idempotency_key": idem_key,
                },
                headers=headers,
            )
    except Exception as e:
        return f"{_DELEGATION_ERROR_PREFIX}dispatch failed: {e}"

    if resp.status_code not in (200, 202):
        return (
            f"{_DELEGATION_ERROR_PREFIX}dispatch returned "
            f"HTTP {resp.status_code}: {resp.text[:200]}"
        )

    try:
        dispatch = resp.json()
    except Exception as e:
        return f"{_DELEGATION_ERROR_PREFIX}non-JSON dispatch response: {e}"

    delegation_id = dispatch.get("delegation_id", "")
    if not delegation_id:
        return f"{_DELEGATION_ERROR_PREFIX}missing delegation_id in response: {dispatch}"

    # Poll for terminal status.
    for attempt in range(_POLL_MAX_ATTEMPTS):
        await asyncio.sleep(_POLL_INTERVAL_S)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                poll_resp = await client.get(
                    f"{_platform_url}/workspaces/{src}/delegations",
                    headers=headers,
                )
        except Exception as e:
            logger.warning("delegate_task poll attempt %s failed: %s", attempt, e)
            continue

        if poll_resp.status_code != 200:
            continue

        try:
            delegations = poll_resp.json()
            if not isinstance(delegations, list):
                delegations = []
        except Exception:
            delegations = []

        for d in delegations:
            if d.get("delegation_id") == delegation_id:
                status = d.get("status", "")
                if status == "completed":
                    return d.get("response_preview", "")
                elif status == "failed":
                    error_detail = d.get("error_detail", "")
                    return f"{_A2A_ERROR_PREFIX}{error_detail or 'delegation failed'}"
                # Any other status: keep polling.

    return f"{_DELEGATION_ERROR_PREFIX}timeout after {_POLL_BUDGET_S}s waiting for delegation {delegation_id}"


# ---------------------------------------------------------------------------
# Tool: delegate_task_async
# ---------------------------------------------------------------------------

TOOL_SCHEMA_DELEGATE_TASK_ASYNC = {
    "name": "delegate_task_async",
    "description": (
        "Fire-and-forget delegation — sends the task and immediately "
        "returns the delegation_id without waiting. Use "
        "check_task_status(workspace_id, delegation_id) to poll for the result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "workspace_id": {
                "type": "string",
                "description": "The target workspace UUID.",
            },
            "task": {
                "type": "string",
                "description": "The task description.",
            },
            "source_workspace_id": {
                "type": "string",
                "description": "Optional source workspace override.",
            },
        },
        "required": ["workspace_id", "task"],
    },
}


async def tool_delegate_task_async(
    workspace_id: str,
    task: str,
    source_workspace_id: str | None = None,
) -> dict[str, Any]:
    """Fire-and-forget delegation — returns delegation_id immediately.

    Returns:
        Dict with delegation_id and source_workspace_id.
    """
    src = source_workspace_id or _workspace_id
    headers = _auth_headers(src)

    idem_key = hashlib.sha256(
        f"{src}:{workspace_id}:{task}".encode()
    ).hexdigest()[:32]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_platform_url}/workspaces/{src}/delegate",
                json={
                    "target_id": workspace_id,
                    "task": task,
                    "idempotency_key": idem_key,
                },
                headers=headers,
            )
    except Exception as e:
        return {
            "ok": False,
            "error": f"dispatch failed: {e}",
            "delegation_id": None,
        }

    if resp.status_code not in (200, 202):
        return {
            "ok": False,
            "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            "delegation_id": None,
        }

    try:
        dispatch = resp.json()
    except Exception:
        dispatch = {}

    return {
        "ok": True,
        "delegation_id": dispatch.get("delegation_id", ""),
        "source_workspace_id": src,
    }


# ---------------------------------------------------------------------------
# Tool: check_task_status
# ---------------------------------------------------------------------------

TOOL_SCHEMA_CHECK_TASK_STATUS = {
    "name": "check_task_status",
    "description": (
        "Poll the platform for a delegation's current status. "
        "Returns the status, response_preview on completed, and "
        "error_detail on failed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "delegation_id": {
                "type": "string",
                "description": "The delegation_id returned by delegate_task_async.",
            },
            "source_workspace_id": {
                "type": "string",
                "description": "Optional source workspace override.",
            },
        },
        "required": ["delegation_id"],
    },
}


async def tool_check_task_status(
    delegation_id: str,
    source_workspace_id: str | None = None,
) -> dict[str, Any]:
    """Poll for delegation status by id.

    Returns:
        Dict with status, response_preview, error_detail.
    """
    src = source_workspace_id or _workspace_id
    headers = _auth_headers(src)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_platform_url}/workspaces/{src}/delegations",
                headers=headers,
            )
    except Exception as e:
        return {"status": "error", "error": f"poll failed: {e}"}

    if resp.status_code != 200:
        return {"status": "error", "error": f"HTTP {resp.status_code}"}

    try:
        delegations = resp.json()
        if not isinstance(delegations, list):
            delegations = []
    except Exception:
        delegations = []

    for d in delegations:
        if d.get("delegation_id") == delegation_id:
            return {
                "status": d.get("status", "unknown"),
                "response_preview": d.get("response_preview", ""),
                "error_detail": d.get("error_detail", ""),
                "created_at": d.get("created_at", ""),
                "updated_at": d.get("updated_at", ""),
            }

    return {
        "status": "not_found",
        "error": f"delegation_id {delegation_id} not found",
    }
