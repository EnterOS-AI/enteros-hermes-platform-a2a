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
