"""Single source of truth for figure styling across the course.

Every figure is produced twice from the same plotting function:

    name.png       dark ink on white, for print and the inline notebook view
    name_dark.png  light ink on a transparent background, for dark presentations

Usage in a notebook::

    from pyutils.plotstyle import set_plot_style, save_dual
    set_plot_style()                      # light style for inline work

    def plot_convergence():
        fig, ax = plt.subplots()
        ax.plot(h, err, marker="o", label="RK4")
        ax.set(xlabel="$h$", ylabel="error"); ax.legend()

    save_dual(plot_convergence, "vessel_rk4_order", FIG)

Design notes
------------
* rcParams bind at *artist creation* time, so the figure must be re-plotted for
  each variant -- hence the callable pattern (a closure over the data), not a
  re-save of an existing figure.
* Fonts are STIXGeneral / mathtext "stix": a Times-compatible face that ships
  inside matplotlib, so figures reproduce identically on any machine without
  needing a system font installed.
* The two palettes are the same hues at two lightnesses -- saturated and light
  for dark backgrounds, darkened for white paper -- so a figure reads the same
  either way.
"""

import os

import matplotlib.pyplot as plt

# --- palettes -----------------------------------------------------------
# Accents for dark backgrounds: saturated and light enough to read on near-black.
COLORS_DARK = {
    "blue": "#7BA0FF",      # edBlue
    "orange": "#FFA870",    # edOrange
    "green": "#34D399",     # edExceeding
    "red": "#F87171",       # edRisk
    "violet": "#B89AFF",    # edViolet
    "yellow": "#FBBF24",    # edWatch
    "text": "#EEEEF5",      # edText
    "muted": "#8888A0",     # edTextSec
}

# Accents for white paper: darkened siblings of the same hues.
COLORS_LIGHT = {
    "blue": "#2A5DB0",      # accentblue
    "orange": "#C25A1E",    # accentorange
    "green": "#1B7A4E",
    "red": "#B03A3A",
    "violet": "#6B4FC8",
    "yellow": "#A87B00",
    "text": "#000000",
    "muted": "#555555",
}

_CYCLE_KEYS = ("blue", "orange", "green", "red", "violet", "yellow")


def _cycle(colors):
    from cycler import cycler
    return cycler(color=[colors[k] for k in _CYCLE_KEYS])


def rc_params(mode="light"):
    """Full rcParams dict for one mode ("light" or "dark")."""
    if mode not in ("light", "dark"):
        raise ValueError(f"mode must be 'light' or 'dark', got {mode!r}")
    colors = COLORS_LIGHT if mode == "light" else COLORS_DARK
    rc = {
        # Times-compatible fonts (matches mathptmx in the LaTeX documents)
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 11,
        # layout
        "figure.figsize": (6, 4),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "image.cmap": "viridis",
        "axes.prop_cycle": _cycle(colors),
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    }
    if mode == "dark":
        fg = colors["text"]
        rc.update({
            "text.color": fg,
            "axes.labelcolor": fg,
            "axes.titlecolor": fg,
            "axes.edgecolor": fg,
            "xtick.color": fg,
            "ytick.color": fg,
            "grid.color": colors["muted"],
            "legend.labelcolor": fg,
            "legend.facecolor": "none",
            "legend.edgecolor": colors["muted"],
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.transparent": True,
        })
    else:
        rc.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.transparent": False,
        })
    return rc


def set_plot_style(mode="light"):
    """Apply the course figure style globally (for inline notebook work)."""
    plt.rcParams.update(rc_params(mode))


def fg():
    """Current foreground ('ink') colour — black in light mode, near-white in
    dark mode.  Use inside a plot function instead of hardcoded 'k'/'black'
    so reference curves stay visible in both variants."""
    return plt.rcParams.get("text.color", "black")


def muted():
    """Current secondary/muted colour, for helper lines and annotations."""
    dark = plt.rcParams.get("savefig.transparent", False)
    return (COLORS_DARK if dark else COLORS_LIGHT)["muted"]


def save_dual(plot_fn, name, figdir, dpi=130, show=True):
    """Render ``plot_fn`` twice and save both variants.

    Writes ``<figdir>/<name>.png`` (light) and ``<figdir>/<name>_dark.png``
    (dark, transparent).  The light figure is left open (and returned) so the
    notebook still displays it inline; the dark one is closed silently.
    """
    os.makedirs(figdir, exist_ok=True)

    # dark first, closed silently
    with plt.rc_context(rc_params("dark")):
        plot_fn()
        fig = plt.gcf()
        fig.savefig(os.path.join(figdir, f"{name}_dark.png"),
                    dpi=dpi, bbox_inches="tight", transparent=True)
        plt.close(fig)

    # light second, left open for inline display
    with plt.rc_context(rc_params("light")):
        plot_fn()
        fig = plt.gcf()
        fig.savefig(os.path.join(figdir, f"{name}.png"),
                    dpi=dpi, bbox_inches="tight")
    if not show:
        plt.close(fig)
    return fig
