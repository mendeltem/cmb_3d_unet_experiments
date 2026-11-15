"""cmbx: 5-fold cross-validation and pre/post-processing experiments for cerebral microbleed (CMB)
detection with the anisotropic 3D U-Net of cmb-valdo-chain (tournament winner ``a03-aniso``).

Python API (cmbx/api.py): ``from cmbx import env, get_model, predict, train, fine_tune``."""
import sys as _sys
import types as _types

__version__ = "0.1.0"


class _Package(_types.ModuleType):
    """``cmbx.train`` and ``cmbx.predict`` stay the API functions even after the submodules of the same name are
    imported (Python would otherwise set the package attribute to the submodule). ``from cmbx.train import ...`` and
    ``import cmbx.predict`` work as before."""
    def __setattr__(self, name, value):
        if name in ("train", "predict") and isinstance(value, _types.ModuleType) and name in self.__dict__:
            return
        super().__setattr__(name, value)


_sys.modules[__name__].__class__ = _Package

from .api import fine_tune, get_model, predict, train  # noqa: E402,F401
from .envfile import env  # noqa: E402,F401
