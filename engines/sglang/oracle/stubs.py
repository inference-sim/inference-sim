# Original to blis-sglang.
"""Stubs for the CUDA-only modules the scheduler imports and the oracle cannot install on CPU.

oracle/requirements.txt leaves SGLang's CUDA-only kernel and runtime packages out of the lock.
A module of one that the scheduler's import or start-up reaches is listed in STUBBED and, with
the reason, in oracle/mocks.json (kind environment); install() makes every name in STUBBED, and
its submodules, import as a module whose every attribute is a MagicMock. That is the stub SGLang's
own CPU tests use for sgl_kernel (python/sglang/test/test_utils.py:2042-2077@297e54fd). A module
not in STUBBED stays absent, so a new import of one fails rather than passing on a stub nobody
listed.
"""
import importlib.abc
import importlib.machinery
import sys
import types

# The top-level modules stubbed, each listed in oracle/mocks.json.
STUBBED: tuple = ()


class _Stub(types.ModuleType):
    """A stubbed module: every attribute is a MagicMock."""

    def __getattr__(self, name):
        from unittest.mock import MagicMock

        return MagicMock()


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in STUBBED:
            return importlib.machinery.ModuleSpec(fullname, _Loader(), is_package=True)
        return None


def install() -> None:
    if STUBBED and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
