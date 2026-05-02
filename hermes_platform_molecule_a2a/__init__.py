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
    ctx.register_platform_adapter(
        name="molecule-a2a",
        adapter_class=MoleculeA2APlatformAdapter,
        requirements_check=check_molecule_a2a_requirements,
    )
