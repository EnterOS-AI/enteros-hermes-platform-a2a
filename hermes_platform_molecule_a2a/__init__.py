"""Hermes plugin: Molecule A2A platform adapter.

Loaded by hermes_cli.plugins via the ``hermes_agent.plugins`` entry
point. The ``register`` callable below is invoked with a PluginContext;
it claims the platform name ``molecule-a2a`` so that GatewayConfig
routes ``platforms.molecule-a2a`` config blocks to this adapter.
"""

from .adapter import MoleculeA2APlatformAdapter, check_molecule_a2a_requirements

__all__ = [
    "MoleculeA2APlatformAdapter",
    "check_molecule_a2a_requirements",
    "register",
]


def register(ctx) -> None:
    """Plugin entry point — dual-mode for upstream + legacy fork APIs.

    Upstream NousResearch/hermes-agent#17751 (merged 2026-04-30) shipped
    ``ctx.register_platform(name, label, adapter_factory, check_fn, ...)``.
    Pre-#17751 forks expose ``ctx.register_platform_adapter(name,
    adapter_class, requirements_check)`` instead — narrower signature,
    no factory. Detect at runtime so the same wheel installs cleanly on
    both.
    """
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
