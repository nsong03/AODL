r"""The documentation runs, and the numbers in it are the code's numbers (WO-18 §4).

Two ways a manual rots, both cheap to prevent:

* **the code moves and the examples stop working** — so every fenced ``python`` block of
  ``README.md`` and ``docs/guide.md`` is executed here, in order, in one namespace per file,
  with the quickstart asserted to produce a real :class:`~aodl.api.MotionPlan`.  A block that
  would cost tens of seconds (a movie render) opts out with a first-line comment and is only
  compiled;
* **a constant is retuned and the prose keeps the old value** — so the guide's closing table
  of quoted numbers is evaluated against :func:`aodl.params.default_1030`, and the four
  canonical constants (10.3 µm per MHz, the 206 µs Eq. 1 budget, the 1.0655 µm waist, the
  11.54 µs transit) are re-checked wherever the prose spells them out.

Deliberately small: this is a rot alarm, not a documentation test suite.  What the numbers
*mean* is tested where they come from.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from aodl.params import AODLParams, default_1030
from aodl.units import MHz, ms, um, us
from aodl.waveform.synthesis import max_z_integral

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
GUIDE = ROOT / "docs" / "guide.md"

#: First line of a fenced block that is too slow to execute here; it is compiled instead, so
#: syntax rot is still caught.  Spelled out in the docs so a reader knows why it is there.
SKIP_MARKER = "# not run by tests/test_docs.py"

#: ``(regex, allowed values)`` for the constants the prose spells out.  The regex captures the
#: number *in the spelling the docs use*, so a stale value fails and a rephrasing shows up as a
#: missing match rather than a silent pass.  Two values are allowed for the axial budget: Eq. 1
#: as written, and the doubled ceiling ``f_z_bias="auto"`` buys.
CANONICAL: tuple[tuple[str, str], ...] = (
    (r"(\d+(?:\.\d+)?) µm per MHz", "P.deflection_scale * MHz / um"),
    (r"(\d+(?:\.\d+)?) µs at 10 µm", "max_z_integral(P) / (10 * um) / us"),
    (r"w₀ = (\d+(?:\.\d+)?) µm", "P.optics.waist0 / um"),
    (r"τ = (\d+(?:\.\d+)?) µs", "P.channels['Ax'].transit_time / us"),
)


def _namespace(params: AODLParams) -> dict[str, Any]:
    """What a documented expression may refer to: the preset, the units, the Eq. 1 ceiling."""
    return {
        "P": params,
        "MHz": MHz,
        "ms": ms,
        "um": um,
        "us": us,
        "max_z_integral": max_z_integral,
    }


def _python_blocks(path: Path) -> list[str]:
    """Every fenced ``python`` block of a markdown file, in document order."""
    return re.findall(r"^```python\n(.*?)^```", path.read_text(encoding="utf-8"), re.M | re.S)


def _run(path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Execute a file's blocks in one namespace, in a scratch directory, and return it."""
    monkeypatch.chdir(tmp_path)
    blocks = _python_blocks(path)
    assert blocks, f"{path.name} has no python examples left to check"
    namespace: dict[str, Any] = {}
    for i, block in enumerate(blocks):
        code = compile(block, f"{path.name}[block {i}]", "exec")
        if not block.lstrip().startswith(SKIP_MARKER):
            exec(code, namespace)  # noqa: S102 - executing the docs is the point
    return namespace


def _quoted_numbers() -> list[tuple[str, str, str]]:
    """``(quantity, value, expression)`` rows of the guide's closing table."""
    rows = []
    for line in GUIDE.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) == 3 and cells[2].startswith("`") and cells[2].endswith("`"):
            rows.append((cells[0], cells[1], cells[2].strip("`")))
    return rows


# ===================================================================== the examples run


def test_the_readme_quickstart_plans_a_move(tmp_path, monkeypatch):
    """The eight lines on the front page produce a plan and the file they claim to write."""
    namespace = _run(README, tmp_path, monkeypatch)

    plan = namespace["plan"]
    assert type(plan).__name__ == "MotionPlan"
    assert plan.report.mode in ("s19", "shepard")
    assert (tmp_path / "move.npz").exists()


def test_the_guide_examples_run(tmp_path, monkeypatch):
    """Every executable block of the manual, in order, sharing one namespace as a reader would."""
    namespace = _run(GUIDE, tmp_path, monkeypatch)

    plan = namespace["plan"]
    assert type(plan).__name__ == "MotionPlan"
    assert plan.report.summary().startswith("AODL motion plan")
    assert (tmp_path / "move.npz").exists()
    # the hardware section really does rebuild the stack it describes
    assert namespace["P"].optics.focal_length == pytest.approx(0.010)
    # and the AWG window is what the text says it is: 2 µs at 625 MS/s, all four channels
    assert set(namespace["window"]) == set(plan.wfs.channels)
    assert all(samples.shape == (1251,) for samples in namespace["window"].values())


# ================================================================== the numbers are current


def test_the_guides_quoted_numbers_are_the_codes_numbers():
    """Each row of the closing table is re-derived, to the precision the table prints."""
    namespace = _namespace(default_1030())
    rows = _quoted_numbers()
    assert len(rows) >= 9, "the guide's table of quoted numbers has lost rows"

    for quantity, value, expression in rows:
        printed = re.match(r"(-?\d+(?:\.\d+)?)", value)
        assert printed is not None, f"{quantity!r}: {value!r} does not start with a number"
        text = printed.group(1)
        decimals = len(text.partition(".")[2])
        computed = round(float(eval(expression, dict(namespace))), decimals)  # noqa: S307
        assert computed == float(text), f"{quantity!r}: doc says {text}, code says {computed}"


@pytest.mark.parametrize("path", [README, GUIDE], ids=lambda p: p.name)
def test_the_canonical_constants_are_not_stale(path):
    """Wherever the prose spells one of the four out, it spells out the current value."""
    params = default_1030()
    namespace = _namespace(params)
    text = path.read_text(encoding="utf-8")

    for pattern, expression in CANONICAL:
        value = float(eval(expression, dict(namespace)))  # noqa: S307
        for found in re.findall(pattern, text):
            decimals = len(found.partition(".")[2])
            allowed = {round(value, decimals), round(2.0 * value, decimals)}
            assert float(found) in allowed, (
                f"{path.name}: {found!r} is stale for {pattern!r}; current value {value}"
            )

    if path is GUIDE:  # the manual must still state all four somewhere
        for pattern, _ in CANONICAL:
            assert re.search(pattern, text), f"the guide no longer states {pattern!r}"


@pytest.mark.parametrize("path", [README, GUIDE], ids=lambda p: p.name)
def test_the_docs_point_at_files_that_exist(path):
    """Relative markdown links resolve — the map is only useful while it is accurate."""
    targets = re.findall(r"\]\((?!https?:)([^)#]+)", path.read_text(encoding="utf-8"))
    assert targets
    for target in targets:
        assert (path.parent / target).resolve().exists(), f"{path.name} links to missing {target}"
