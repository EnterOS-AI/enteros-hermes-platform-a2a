"""Tests for delegation_tools — list_peers, delegate_task, delegate_task_async,
check_task_status.

These tests mock httpx at the client level so they run without a live
platform. The test patterns mirror molecule-runtime's test coverage
for the same tool surface (see workspace/tests/test_a2a_tools_delegation.py).

Import delegation_tools via importlib to avoid pulling in the
gateway/hermes dependencies that __init__.py transitively loads.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import pytest

# Load delegation_tools by file path so the package __init__.py
# (which imports gateway) is never executed.
PKG_ROOT = Path(__file__).resolve().parent.parent
_DT_PATH = PKG_ROOT / "hermes_platform_molecule_a2a" / "delegation_tools.py"
spec = importlib.util.spec_from_file_location("delegation_tools", _DT_PATH)
_dt_mod = importlib.util.module_from_spec(spec)
sys.modules["delegation_tools"] = _dt_mod
spec.loader.exec_module(_dt_mod)  # type: ignore[attr-defined]
dt = _dt_mod

# ---------------------------------------------------------------------------
# Dummy response — mirrors the httpx.Response shape we actually use.
# ---------------------------------------------------------------------------


class DummyResponse:
    def __init__(self, status: int, json_data: Any):
        self.status_code = status  # mirrors httpx.Response attribute name
        self.status = status        # also available as .status (backward compat)
        self._json = json_data
        # .text is used for error messages: resp.text[:200]
        self._text = str(json_data) if not isinstance(json_data, str) else json_data

    def json(self) -> Any:
        return self._json

    @property
    def text(self) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Mock httpx client factory.
# Returns a context-manager that yields a client whose get/post methods
# return the provided DummyResponse.
# ---------------------------------------------------------------------------


def make_httpx_mock(
    get_resp: Optional[DummyResponse] = None,
    post_resp: Optional[DummyResponse] = None,
):
    """Create an httpx.AsyncClient patch that returns fixed responses.

    The patch replaces httpx.AsyncClient at the module level used by
    delegation_tools. Since the replacement is a plain class, not a
    coroutine, we implement the async context manager on the class itself
    so `async with httpx.AsyncClient(...) as client:` works correctly.
    """
    dummy_get = get_resp or DummyResponse(200, [])
    dummy_post = post_resp or DummyResponse(200, {})

    class FakeClient:
        async def get(self, url: str, **kwargs):
            return dummy_get

        async def post(self, url: str, json: dict = None, **kwargs):
            return dummy_post

    class PatchedAsyncClient:
        """Replaces httpx.AsyncClient in delegation_tools.

        The real httpx.AsyncClient.__aenter__ is a coroutine (async def).
        PatchedAsyncClient must mirror that so `async with X as c` works
        in Python 3.11+ (Python raises TypeError if __aenter__ is not
        awaitable on a class used in `async with`).
        """

        def __init__(self, *args, **kwargs):
            pass  # absorb any positional args from httpx call sites

        async def __aenter__(self) -> "FakeClient":
            return FakeClient()

        async def __aexit__(self, *args) -> None:
            pass

    return patch("httpx.AsyncClient", PatchedAsyncClient)


# ---- helpers --------------------------------------------------------


async def mock_get(url: str, **kwargs) -> DummyResponse:
    """Route GET to the right mock response by URL path."""
    if "/delegations" in url:
        return DummyResponse(200, [
            {
                "delegation_id": "did-123",
                "status": "completed",
                "response_preview": "Done",
                "created_at": "2026-05-10T00:00:00Z",
                "updated_at": "2026-05-10T00:01:00Z",
            }
        ])
    return DummyResponse(200, [
        {"id": "peer-1", "name": "ops-agent", "role": "sre", "status": "online"},
        {"id": "peer-2", "name": "pm-agent", "role": "pm", "status": "online"},
    ])


async def mock_post(url: str, json: dict = None, **kwargs) -> DummyResponse:
    """Route POST to the right mock response by URL path."""
    if "/delegate" in url:
        return DummyResponse(202, {"delegation_id": "did-123", "status": "dispatched"})
    return DummyResponse(404, {"error": "not found"})


# ---- configure() ----------------------------------------------------


def test_configure_sets_platform_url_and_workspace_id():
    """configure() must update module-level _platform_url and _workspace_id."""
    dt.configure(platform_url="http://custom:9000", workspace_id="custom-ws")
    assert dt._platform_url == "http://custom:9000"
    assert dt._workspace_id == "custom-ws"


def test_configure_idempotent():
    """Calling configure() twice must not raise."""
    dt.configure(platform_url="http://first:8080", workspace_id="ws-1")
    dt.configure(platform_url="http://second:9090", workspace_id="ws-2")
    assert dt._platform_url == "http://second:9090"
    assert dt._workspace_id == "ws-2"


# ---- tool_list_peers -------------------------------------------------


@pytest.mark.asyncio
async def test_list_peers_returns_peers_on_200():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")
    with make_httpx_mock(get_resp=DummyResponse(200, [
        {"id": "peer-1", "name": "ops-agent", "role": "sre", "status": "online"},
        {"id": "peer-2", "name": "pm-agent", "role": "pm", "status": "online"},
    ])):
        result = await dt.tool_list_peers()
        assert len(result) == 2
        assert result[0]["id"] == "peer-1"
        assert result[0]["name"] == "ops-agent"


@pytest.mark.asyncio
async def test_list_peers_returns_empty_on_non_200():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")
    with make_httpx_mock(get_resp=DummyResponse(503, {})):
        result = await dt.tool_list_peers()
        assert result == []


@pytest.mark.asyncio
async def test_list_peers_returns_empty_on_exception():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")

    class ExplodingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            raise Exception("network error")

    with patch("httpx.AsyncClient", side_effect=lambda *a, **k: ExplodingClient()):
        result = await dt.tool_list_peers()
        assert result == []


# ---- tool_delegate_task ----------------------------------------------


@pytest.mark.asyncio
async def test_delegate_task_returns_response_on_completed():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")
    with make_httpx_mock(
        post_resp=DummyResponse(202, {"delegation_id": "did-123"}),
        get_resp=DummyResponse(200, [
            {
                "delegation_id": "did-123",
                "status": "completed",
                "response_preview": "Task completed successfully",
                "error_detail": "",
            }
        ]),
    ):
        result = await dt.tool_delegate_task("peer-1", "do the thing")
        assert result == "Task completed successfully"


@pytest.mark.asyncio
async def test_delegate_task_returns_error_on_failed():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")
    with make_httpx_mock(
        post_resp=DummyResponse(202, {"delegation_id": "did-456"}),
        get_resp=DummyResponse(200, [
            {
                "delegation_id": "did-456",
                "status": "failed",
                "error_detail": "target workspace unavailable",
                "response_preview": "",
            }
        ]),
    ):
        result = await dt.tool_delegate_task("peer-1", "do the thing")
        assert result.startswith(dt._A2A_ERROR_PREFIX)
        assert "target workspace unavailable" in result


@pytest.mark.asyncio
async def test_delegate_task_returns_error_on_dispatch_failure():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")
    with make_httpx_mock(post_resp=DummyResponse(500, {"error": "server error"})):
        result = await dt.tool_delegate_task("peer-1", "do the thing")
        assert result.startswith(dt._DELEGATION_ERROR_PREFIX)
        assert "500" in result


# ---- tool_delegate_task_async -----------------------------------------


@pytest.mark.asyncio
async def test_delegate_task_async_returns_delegation_id():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")
    with make_httpx_mock(post_resp=DummyResponse(202, {"delegation_id": "did-789"})):
        result = await dt.tool_delegate_task_async("peer-2", "quick task")
        assert result["ok"] is True
        assert result["delegation_id"] == "did-789"
        assert result["source_workspace_id"] == "ws-root"


@pytest.mark.asyncio
async def test_delegate_task_async_returns_error_on_failure():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")
    with make_httpx_mock(post_resp=DummyResponse(500, {})):
        result = await dt.tool_delegate_task_async("peer-2", "task")
        assert result["ok"] is False
        assert result["delegation_id"] is None


# ---- tool_check_task_status ------------------------------------------


@pytest.mark.asyncio
async def test_check_task_status_returns_completed_status():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")
    with make_httpx_mock(get_resp=DummyResponse(200, [
        {
            "delegation_id": "did-123",
            "status": "completed",
            "response_preview": "Result text",
            "error_detail": "",
            "created_at": "2026-05-10T00:00:00Z",
            "updated_at": "2026-05-10T00:01:00Z",
        }
    ])):
        result = await dt.tool_check_task_status("did-123")
        assert result["status"] == "completed"
        assert result["response_preview"] == "Result text"


@pytest.mark.asyncio
async def test_check_task_status_returns_not_found():
    dt.configure(platform_url="http://platform:8080", workspace_id="ws-root")
    with make_httpx_mock(get_resp=DummyResponse(200, [])):
        result = await dt.tool_check_task_status("unknown-id")
        assert result["status"] == "not_found"


# ---- tool schemas ---------------------------------------------------


def test_list_peers_schema_is_valid():
    schema = dt.TOOL_SCHEMA_LIST_PEERS
    assert schema["name"] == "list_peers"
    assert "description" in schema
    assert schema["parameters"]["type"] == "object"
    assert schema["parameters"]["required"] == []


def test_delegate_task_schema_is_valid():
    schema = dt.TOOL_SCHEMA_DELEGATE_TASK
    assert schema["name"] == "delegate_task"
    assert "workspace_id" in schema["parameters"]["properties"]
    assert "task" in schema["parameters"]["properties"]
    assert "source_workspace_id" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["workspace_id", "task"]


def test_delegate_task_async_schema_is_valid():
    schema = dt.TOOL_SCHEMA_DELEGATE_TASK_ASYNC
    assert schema["name"] == "delegate_task_async"
    assert schema["parameters"]["required"] == ["workspace_id", "task"]


def test_check_task_status_schema_is_valid():
    schema = dt.TOOL_SCHEMA_CHECK_TASK_STATUS
    assert schema["name"] == "check_task_status"
    assert "delegation_id" in schema["parameters"]["properties"]
    assert schema["parameters"]["required"] == ["delegation_id"]
