"""Make hermes-agent importable for tests.

The plugin imports `gateway.config` and `hermes_cli.plugins` directly.
Those live in the local hermes fork at ~/.hermes/hermes-agent (the
checkout that hosts the patched ``feat/platform-adapter-plugins``
branch). Tests inject that on sys.path so the adapter resolves the
real BasePlatformAdapter contract — no mock substitutes here, since
the whole point is to exercise the real subclass relationship.
"""

import os
import sys
from pathlib import Path

HERMES_REPO = Path.home() / ".hermes" / "hermes-agent"
if HERMES_REPO.exists():
    sys.path.insert(0, str(HERMES_REPO))

# Make the package itself importable when tests are run from the repo
# root without a prior `pip install -e .`.
PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))

# Hermes plugin discovery uses HERMES_HOME for the user plugin dir.
# Point it at a scratch location so test runs don't pick up the
# user's real ~/.hermes/.
os.environ.setdefault(
    "HERMES_HOME", str(PKG_ROOT / "tests" / ".hermes_test_home")
)

import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _platform_registered_like_the_gateway():
    """Register molecule-a2a the way the gateway does when it loads the plugin.

    Upstream hermes (#17751+) only resolves ``Platform("molecule-a2a")`` once
    the name is in its platform registry, which ``ctx.register_platform``
    fills at plugin load. Tests construct the adapter directly, so without
    this every adapter test fails with "'molecule-a2a' is not a valid
    Platform". The retired fork has no registry and needs nothing.
    """
    try:
        from gateway.platform_registry import PlatformEntry, platform_registry
    except ImportError:
        yield
        return
    from hermes_platform_molecule_a2a.adapter import (
        PLATFORM_NAME,
        MoleculeA2APlatformAdapter,
        check_molecule_a2a_requirements,
    )

    if not platform_registry.is_registered(PLATFORM_NAME):
        platform_registry.register(PlatformEntry(
            name=PLATFORM_NAME,
            label="Molecule A2A",
            adapter_factory=MoleculeA2APlatformAdapter,
            check_fn=check_molecule_a2a_requirements,
        ))
    yield
