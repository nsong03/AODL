"""The import discipline of ``aodl_advanced`` (W2 §1, DESIGN §1.5).

The FFT engine may import from ``aodl`` only the parameters, units, the sign authority, the
waveform IR/envelopes/renderer and the polynomial class.  Never ``aodl.field``,
``aodl.device.aod*``, ``aodl.device.mixing``, ``aodl.engine``, ``aodl.api``, ``aodl.viz`` or
``aodl.check`` — otherwise the cross-engine agreement would not be a cross-check.
"""

from __future__ import annotations

import ast
from pathlib import Path

import aodl_advanced

ALLOWED = {
    "aodl.params",
    "aodl.units",
    "aodl.device.conventions",
    "aodl.waveform.tones",
    "aodl.waveform.shepard",
    "aodl.waveform.export",
    "aodl.poly",
}

PACKAGE = Path(aodl_advanced.__file__).resolve().parent


def _aodl_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names if alias.name.split(".")[0] == "aodl")
        elif (
            isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "aodl"
        ):
            # ``from aodl.device import conventions`` names the module through its attribute.
            for alias in node.names:
                candidate = f"{node.module}.{alias.name}"
                found.append(candidate if candidate in ALLOWED else node.module)
    return found


def test_every_module_is_scanned() -> None:
    modules = sorted(p.name for p in PACKAGE.glob("*.py"))
    assert {"efficiency.py", "pupil.py", "propagate.py", "engine.py", "schemas.py"} <= set(modules)


def test_only_allowed_aodl_modules_are_imported() -> None:
    offenders: dict[str, list[str]] = {}
    for path in PACKAGE.rglob("*.py"):
        bad = [name for name in _aodl_imports(path) if name not in ALLOWED]
        if bad:
            offenders[path.name] = bad
    assert not offenders, f"forbidden aodl imports: {offenders}"


def test_bare_aodl_import_is_forbidden() -> None:
    """``import aodl`` would pull in ``aodl.api`` / ``aodl.engine`` through the package init."""
    for path in PACKAGE.rglob("*.py"):
        assert "aodl" not in _aodl_imports(path), path.name
