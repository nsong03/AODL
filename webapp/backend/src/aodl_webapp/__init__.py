"""AODL Studio backend — the HTTP face of the ``aodl`` package.

Deliberately bare: this module carries the version string and nothing else.  The route
modules (including W2's ``routes/advanced.py``) import ``aodl_webapp.planner``,
``aodl_webapp.simulation``, ``aodl_webapp.schemas`` and ``aodl_webapp.errors`` directly, and a
package-level import of :mod:`aodl_webapp.main` here would close that loop into a cycle.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
