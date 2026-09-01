r"""The M6 independence rule, enforced by scanning ``src/aodl/check/``'s own source.

The whole value of the checker is that it does **not** share the simulator's machinery.  If
``check/`` ever imported ``field/focal.py`` or ``device/aodl.py``, a sign error in either
would cancel out of the comparison and every M6 verdict would be worth nothing.  So the rule
is mechanical: ``check/`` may import ``aodl.params``, ``aodl.units``, ``aodl.poly``,
``aodl.trajectory.spec``, ``aodl.device.conventions``, named constants from
``aodl.waveform.export``, its own submodules, numpy/scipy and the standard library.  Nothing
else.

*Why a source scan and not an import probe.*  ``import aodl`` pulls in ``api`` -> ``engine``
-> ``field`` -> ``device``, so by the time any ``aodl.check`` module is importable the
forbidden modules are already in ``sys.modules``; a subprocess that watched ``sys.modules``
could prove nothing.  The scan reads the files instead.  It uses :mod:`ast` rather than a
regular expression — same idea, but it cannot be fooled by an import written inside a
docstring, and it sees ``import a.b as c`` and ``from . import x`` correctly.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import aodl.check
from aodl.waveform import export

#: Modules inside the package that ``check/`` may import.
ALLOWED_AODL_MODULES = frozenset(
    {
        "aodl.params",
        "aodl.units",
        "aodl.poly",
        "aodl.trajectory.spec",
        "aodl.device.conventions",
        "aodl.waveform.export",
    }
)

#: Names ``check/`` may take from ``aodl.waveform.export``: constants only.  Importing the
#: module's *functions* would mean the checker re-rendered samples through the very code path
#: it is meant to audit.
ALLOWED_EXPORT_NAMES = frozenset(
    {"DEFAULT_SAMPLE_RATE", "SAMPLES_SCHEMA_VERSION", "SAMPLES_SUFFIX"}
)

#: Modules whose presence would defeat the whole exercise (a subset of everything not
#: allowed, listed explicitly so a failure names the offence).
FORBIDDEN_AODL_MODULES = frozenset(
    {
        "aodl.field",
        "aodl.field.focal",
        "aodl.field.gaussian",
        "aodl.field.measure",
        "aodl.field.reference",
        "aodl.device.aod",
        "aodl.device.aodl",
        "aodl.device.mixing",
        "aodl.engine",
        "aodl.api",
        "aodl.viz",
        "aodl.viz.movie",
        "aodl.waveform.tones",
        "aodl.waveform.synthesis",
        "aodl.waveform.shepard",
        "aodl.waveform.serialize",
    }
)

#: Third-party roots the checker may use.
ALLOWED_THIRD_PARTY = frozenset({"numpy", "scipy"})

#: Standard-library modules the checker may use.
ALLOWED_STDLIB = frozenset(
    {"__future__", "ast", "collections", "dataclasses", "json", "math", "pathlib", "typing"}
)

PACKAGE_DIR = Path(aodl.check.__file__).parent
PACKAGE_ROOT = PACKAGE_DIR.parent


def _sources() -> list[Path]:
    files = sorted(PACKAGE_DIR.glob("*.py"))
    assert files, f"no sources found under {PACKAGE_DIR}"
    return files


def _resolve(module: str | None, level: int) -> str:
    """Absolute dotted name of an import target, resolving ``from .x import y``."""
    if level == 0:
        return module or ""
    # ``check`` sits at aodl.check; level 1 is the package itself, level 2 is ``aodl``.
    base = ["aodl", "check"][: 2 - (level - 1)]
    return ".".join([*base, *([module] if module else [])])


def _is_module(dotted: str) -> bool:
    """Is this dotted name a module of *this* package's source tree?  (No import needed.)"""
    parts = dotted.split(".")
    if parts[0] != "aodl":
        return False
    base = PACKAGE_ROOT.joinpath(*parts[1:])
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def _imports(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """``[(module, imported names), ...]`` for every import statement in ``path``.

    ``from P import m`` where ``P.m`` is itself a module counts as importing ``P.m`` — that
    is what ``from ..device import conventions`` does, and the intermediate package is only
    a namespace (:func:`test_the_intermediate_packages_are_only_namespaces`).
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            target = _resolve(node.module, node.level)
            names = tuple(alias.name for alias in node.names)
            submodules = tuple(name for name in names if _is_module(f"{target}.{name}"))
            found.extend((f"{target}.{name}", ()) for name in submodules)
            attributes = tuple(name for name in names if name not in submodules)
            if attributes or not submodules:
                found.append((target, attributes))
    return found


def _root(module: str) -> str:
    return module.split(".", 1)[0]


def test_the_scan_finds_every_source_file() -> None:
    """The guard is only worth anything if it is looking at all of ``check/``."""
    names = {path.name for path in _sources()}
    assert names == {
        "__init__.py",
        "demod.py",
        "metrics.py",
        "pupil.py",
        "record.py",
        "transform.py",
    }


def test_check_imports_only_the_allowlist() -> None:
    """Every import statement in ``src/aodl/check/*.py``, checked against the rule."""
    offences: list[str] = []
    for path in _sources():
        for module, _ in _imports(path):
            if module.startswith("aodl.check") or module == "aodl.check":
                continue  # its own submodules
            root = _root(module)
            if root == "aodl":
                if module not in ALLOWED_AODL_MODULES:
                    offences.append(f"{path.name}: imports {module!r}")
            elif root in ALLOWED_THIRD_PARTY:
                continue
            elif root in ALLOWED_STDLIB:
                continue
            else:
                offences.append(f"{path.name}: imports {module!r} (not on the allowlist)")
    assert not offences, "src/aodl/check/ broke the M6 independence rule:\n  " + "\n  ".join(
        offences
    )


def test_the_forbidden_modules_are_named_nowhere_in_check() -> None:
    """Belt and braces: the simulator's module names do not appear in the sources at all.

    An import is the way it would happen, but a deferred ``importlib.import_module`` or a
    ``sys.modules`` lookup would not be an ``ast.Import`` node — so the text is checked too.
    """
    for path in _sources():
        text = path.read_text()
        code = "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith(("#", "*", ":"))
        )
        for banned in ("importlib", "__import__", "sys.modules", "eval(", "exec("):
            assert banned not in code, f"{path.name} reaches for {banned!r}"
        for module in FORBIDDEN_AODL_MODULES:
            # The module names may be *mentioned* in prose; what must not appear is an import.
            assert f"import {module}" not in text, f"{path.name} imports {module}"
            assert f"from {module}" not in text, f"{path.name} imports from {module}"


def test_only_constants_are_taken_from_the_export_module() -> None:
    """``waveform/export`` is on the allowlist for its schema constants, not its renderer."""
    taken: set[str] = set()
    for path in _sources():
        for module, names in _imports(path):
            if module == "aodl.waveform.export":
                taken.update(names)
    assert taken, "the samples-file schema constants should come from waveform/export"
    assert taken <= ALLOWED_EXPORT_NAMES, f"non-constant names imported from export: {taken}"
    # ... and they really are constants of that module, not stale copies.
    for name in taken:
        assert hasattr(export, name)
        assert not callable(getattr(export, name))


def test_relative_imports_resolve_to_the_names_they_claim() -> None:
    """Sanity check on the resolver the scan relies on."""
    assert _resolve("params", 2) == "aodl.params"
    assert _resolve("device.conventions", 2) == "aodl.device.conventions"
    assert _resolve("record", 1) == "aodl.check.record"
    assert _resolve(None, 1) == "aodl.check"
    assert _is_module("aodl.device.conventions")
    assert _is_module("aodl.device")
    assert not _is_module("aodl.device.geometry")  # a name, not a module
    assert not _is_module("numpy")


def test_the_intermediate_packages_are_only_namespaces() -> None:
    """``from ..device import conventions`` must not drag a sibling in through ``__init__``.

    The scan credits that statement with importing ``aodl.device.conventions`` alone, which is
    only true while ``aodl/device/__init__.py`` re-exports nothing.  Same for the other
    subpackages the allowlist reaches through.
    """
    for package in ("device", "waveform", "trajectory"):
        tree = ast.parse((PACKAGE_ROOT / package / "__init__.py").read_text())
        body = [node for node in tree.body if not isinstance(node, ast.Expr)]
        assert body == [], f"aodl/{package}/__init__.py is no longer a bare namespace"


@pytest.mark.parametrize("module", sorted(FORBIDDEN_AODL_MODULES))
def test_the_forbidden_modules_exist_so_the_rule_is_not_vacuous(module: str) -> None:
    """Each name on the forbidden list is a real module of this package."""
    __import__(module)
    assert module in sys.modules
