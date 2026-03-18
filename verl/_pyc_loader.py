from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path


def _cache_tags() -> list[str]:
    tags: list[str] = []
    cache_tag = sys.implementation.cache_tag
    if cache_tag:
        tags.append(cache_tag)
    major = sys.version_info.major
    minor = sys.version_info.minor
    fallback = f"cpython-{major}{minor}"
    if fallback not in tags:
        tags.append(fallback)
    return tags


def _find_pyc(package_root: Path, fullname: str):
    parts = fullname.split(".")
    if not parts:
        return None, False

    rel_parts = parts[1:]
    target_dir = package_root.joinpath(*rel_parts)

    for tag in _cache_tags():
        package_pyc = target_dir / "__pycache__" / f"__init__.{tag}.pyc"
        if package_pyc.exists():
            return package_pyc, True

    if not rel_parts:
        module_base = package_root
        stem = "__init__"
    else:
        module_base = package_root.joinpath(*rel_parts[:-1])
        stem = rel_parts[-1]

    for tag in _cache_tags():
        module_pyc = module_base / "__pycache__" / f"{stem}.{tag}.pyc"
        if module_pyc.exists():
            return module_pyc, False

    return None, False


class _PycCacheFinder(importlib.abc.MetaPathFinder):
    def __init__(self, package_name: str, package_root: Path):
        self.package_name = package_name
        self.package_root = package_root

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname == self.package_name:
            return None
        if not fullname.startswith(f"{self.package_name}."):
            return None

        pyc_path, is_package = _find_pyc(self.package_root, fullname)
        if pyc_path is None:
            return None

        loader = importlib.machinery.SourcelessFileLoader(fullname, str(pyc_path))
        if is_package:
            return importlib.util.spec_from_file_location(
                fullname,
                pyc_path,
                loader=loader,
                submodule_search_locations=[str(pyc_path.parent.parent)],
            )
        return importlib.util.spec_from_file_location(fullname, pyc_path, loader=loader)


def install_pyc_finder(package_name: str, package_root: Path) -> None:
    for finder in sys.meta_path:
        if (
            isinstance(finder, _PycCacheFinder)
            and finder.package_name == package_name
            and finder.package_root == package_root
        ):
            return
    sys.meta_path.insert(0, _PycCacheFinder(package_name=package_name, package_root=package_root))


def exec_pyc_into_globals(fullname: str, package_root: Path, module_globals: dict) -> None:
    pyc_path, _ = _find_pyc(package_root, fullname)
    if pyc_path is None:
        raise ImportError(f"Could not find compiled module for {fullname} under {package_root}")

    loader = importlib.machinery.SourcelessFileLoader(fullname, str(pyc_path))
    code = loader.get_code(fullname)
    if code is None:
        raise ImportError(f"Could not load code object from {pyc_path}")
    exec(code, module_globals)
