from pathlib import Path

from ._pyc_loader import install_pyc_finder


_PACKAGE_ROOT = Path(__file__).resolve().parent

# The repository ships most of `verl` as pyc-only modules under `__pycache__`.
# Register a finder once so normal imports like `verl.protocol` keep working.
install_pyc_finder(__name__, _PACKAGE_ROOT)

# Current recipes only rely on the package root to expose `DataProto`.
from .protocol import DataProto

__all__ = ["DataProto"]
