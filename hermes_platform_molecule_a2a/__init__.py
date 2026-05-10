"""Hermes plugin: Molecule A2A platform adapter.

Loaded by hermes_cli.plugins via the ``hermes_agent.plugins`` entry
point. The ``register`` callable below is invoked with a PluginContext;
it claims the platform name ``molecule-a2a`` so that GatewayConfig
routes ``platforms.molecule-a2a`` config blocks to this adapter,
and registers the list_peers / delegate_task family of tools so the
hermes agent can use them without needing the platform runtime's MCP.

Delegation tools registered:
  - ``list_peers``          — query the platform registry for peer workspaces
  - ``delegate_task``       — sync delegation (platform async polling path)
  - ``delegate_task_async``  — fire-and-forget delegation (returns delegation_id)
  - ``check_task_status``   — poll delegation status by id

These are registered via ``ctx.register_tool()`` at hermes boot so the
hermes LLM session can call them the same way it calls built-in hermes
tools (search, send_message, …). See issue #157.
"""

from __future__ import annotations

from .adapter import MoleculeA2APlatformAdapter, check_molecule_a2a_requirements
from .delegation_tools import (
    TOOL_SCHEMA_CHECK_TASK_STATUS,
    TOOL_SCHEMA_DELEGATE_TASK,
    TOOL_SCHEMA_DELEGATE_TASK_ASYNC,
    TOOL_SCHEMA_LIST_PEERS,
    tool_check_task_status,
    tool_delegate_task,
    tool_delegate_task_async,
    tool_list_peers,
)

__all__ = [
    "MoleculeA2APlatformAdapter",
    "check_molecule_a2a_requirements",
    "register",
    # Delegation tools (exported so tests can import without reaching
    # into the sub-module).
    "tool_list_peers",
    "tool_delegate_task",
    "tool_delegate_task_async",
    "tool_check_task_status",
    "TOOL_SCHEMA_LIST_PEERS",
    "TOOL_SCHEMA_DELEGATE_TASK",
    "TOOL_SCHEMA_DELEGATE_TASK_ASYNC",
    "TOOL_SCHEMA_CHECK_TASK_STATUS",
]


def register(ctx) -> None:
    """Plugin entry point — dual-mode for upstream + legacy fork APIs.

    Upstream NousResearch/hermes-agent#17751 (merged 2026-04-30) shipped
    ``ctx.register_platform(name, label, adapter_factory, check_fn, ...)``.
    Pre-#17751 forks expose ``ctx.register_platform_adapter(name,
    adapter_class, requirements_check)`` instead — narrower signature,
    no factory. Detect at runtime so the same wheel installs cleanly on
    both.

    Also registers the list_peers / delegate_task family of tools via
    ``ctx.register_tool()`` so hermes can call them as first-class
    agent tools. The tool schemas are defined in ``delegation_tools.py``.
    """
    # Platform adapter registration (existing behavior).
    if hasattr(ctx, "register_platform"):
        ctx.register_platform(
            name="molecule-a2a",
            label="Molecule A2A",
            adapter_factory=lambda cfg: MoleculeA2APlatformAdapter(cfg),
            check_fn=check_molecule_a2a_requirements,
            install_hint=(
                "configure platforms.molecule-a2a in config.yaml; "
                "in-container hermes only — for external hermes use "
                "Molecule-AI/hermes-channel-molecule instead"
            ),
        )
    elif hasattr(ctx, "register_platform_adapter"):
        ctx.register_platform_adapter(
            name="molecule-a2a",
            adapter_class=MoleculeA2APlatformAdapter,
            requirements_check=check_molecule_a2a_requirements,
        )
    else:
        raise RuntimeError(
            "hermes-platform-molecule-a2a: this hermes-agent version "
            "exposes neither register_platform (upstream #17751+) nor "
            "register_platform_adapter (legacy fork) — cannot register"
        )

    # Delegation tools registration.
    # Hermes calls ctx.register_tool() for each tool at boot. The tools
    # are async functions that take typed JSON parameters matching the
    # schema. If register_tool is absent (pre-tool-API hermes versions),
    # log a warning but don't fail the plugin — the adapter still works.
    if hasattr(ctx, "register_tool"):
        tools = [
            (TOOL_SCHEMA_LIST_PEERS, tool_list_peers),
            (TOOL_SCHEMA_DELEGATE_TASK, tool_delegate_task),
            (TOOL_SCHEMA_DELEGATE_TASK_ASYNC, tool_delegate_task_async),
            (TOOL_SCHEMA_CHECK_TASK_STATUS, tool_check_task_status),
        ]
        for schema, handler in tools:
            try:
                ctx.register_tool(schema, handler)
            except Exception as exc:
                # Log and continue — plugin should not crash hermes boot
                # if tool registration fails on a specific tool.
                import logging
                logging.getLogger(__name__).warning(
                    "failed to register tool %s: %s", schema["name"], exc
                )
    else:
        import logging
        logging.getLogger(__name__).warning(
            "hermes-platform-molecule-a2a: ctx.register_tool not available "
            "(hermes version may pre-date the tool API); list_peers and "
            "delegate_task will not be registered as hermes tools"
        )
