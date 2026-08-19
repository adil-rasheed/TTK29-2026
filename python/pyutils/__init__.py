"""Shared helpers for the course notebooks.

Import from any module notebook, e.g.::

    import sys; sys.path.append("..")   # if needed
    from pyutils import set_plot_style

Keep cross-module utilities here (plot styling matched to the slide palette,
ODE/PDE solvers, data loaders) so notebooks import rather than copy-paste.
"""

# Styling lives in pyutils.plotstyle (single source of truth); these re-exports
# keep every existing notebook import working.
from .plotstyle import (  # noqa: F401
    COLORS_DARK,
    COLORS_LIGHT,
    fg,
    muted,
    rc_params,
    save_dual,
    set_plot_style,
)
