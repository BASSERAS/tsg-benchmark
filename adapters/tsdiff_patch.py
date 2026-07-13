"""Patch for gluonts module path changes to support TSDiff."""
import importlib
import sys

# Old paths -> new paths for gluonts >= 0.15
_MODULE_ALIASES = {
    "gluonts.torch.modules.scaler": "gluonts.torch.scaler",
    "gluonts.torch.modules.distribution_output": "gluonts.torch.distributions",
}

# Names to forward from old modules
_EXTRA_NAMES = {
    "gluonts.torch.modules.scaler": ["MeanScaler", "NOPScaler", "StdScaler"],
}


def _install_patches():
    for old_path, new_path in _MODULE_ALIASES.items():
        if old_path in sys.modules:
            continue
        try:
            new_mod = importlib.import_module(new_path)
        except ImportError:
            continue

        # Create a fake module for the old path
        class _ProxyModule:
            def __init__(self, target):
                self.__dict__["_target"] = target
                self.__path__ = []

            def __getattr__(self, name):
                return getattr(self._target, name)

            def __setattr__(self, name, value):
                setattr(self._target, name, value)

        proxy = _ProxyModule(new_mod)
        sys.modules[old_path] = proxy


_install_patches()
