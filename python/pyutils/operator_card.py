"""Operator cards: one figure that shows, for an operator-learning example,
exactly what the branch receives, what the trunk receives, and what comes out.

Three columns, always in the same colours as the DeepONet lecture:

    left    "what the branch sees"   the input function(s) with the sensor dots
                                     (blue) and the resulting vector printed as a
                                     strip of numbers;
    middle  "what the trunk sees"    the query domain with ONE query marked
                                     (orange), the output drawn faintly behind it;
    right   "what comes out"         the output function(s) (violet) with the one
                                     value at the query circled.

A dimension strip under the panels states the vector lengths.

Usage (see python/25-deep-operator-learning/notebooks/don_operator_cards.py)::

    from pyutils.operator_card import operator_card
    def plot():
        operator_card(inputs=[dict(x=xf, y=a_fine, sx=XS, sy=a_sens, label='$a(x)$',
                                   xlabel='$x$', name='initial profile')],
                      query=dict(kind='1d', q=0.3, xlabel='$y$'),
                      outputs=[dict(x=XQ, y=T, label='$T(y,t_f)$')],
                      dims='branch: 40 numbers in ... ')
    save_dual(plot, 'don_heat_card', FIG)

The function draws into the *current* figure (it creates one), so it composes
with ``save_dual``'s callable pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .plotstyle import fg, muted, COLORS_DARK, COLORS_LIGHT


def _colors():
    dark = plt.rcParams.get("savefig.transparent", False)
    return COLORS_DARK if dark else COLORS_LIGHT


def _vector_strip(vals, n_show=3):
    vals = np.asarray(vals).ravel()
    if len(vals) <= 2 * n_show:
        body = ", ".join(f"{v:+.2f}" for v in vals)
    else:
        head = ", ".join(f"{v:+.2f}" for v in vals[:n_show])
        tail = ", ".join(f"{v:+.2f}" for v in vals[-n_show:])
        body = f"{head}, …, {tail}"
    return f"[{body}]"


def operator_card(inputs, query, outputs, dims, title=None, figsize=(13.5, 4.2)):
    """Draw an operator card into a new figure.

    inputs : list of dicts, one per input function (1 or 2):
        x, y        fine curve of the input function
        sx, sy      sensor locations and the readings there
        label       legend label of the function (LaTeX ok)
        xlabel      axis label
        name        short word used in the vector strip, e.g. 'thrust'
        symbol      symbol of the sensor vector, e.g. r'$\\boldsymbol{a}$'
        unit        optional unit for the y axis
    query : dict
        kind        '1d' (query is one coordinate) or '2d' (query is (x,t))
        q           the query value (float, or (x,t) for '2d')
        xlabel      axis label of the query coordinate ('1d')
        ylabel      optional
        -- for '2d': field (2-D array T[t, x]), xs, ts (axes), colloc (N,2) points,
           input_label (text for the t=0 edge), xlabel, tlabel
        -- optional 'modes': list of (x, phi_k, label) to draw instead of the output
           (POD trunk), with 'modes_title'
    outputs : list of dicts, one per output channel (1 to 3):
        x, y        the output curve; label; ylabel optional
        qx          query coordinate at which to mark the value (defaults to query q)
    dims : str      one line printed under the panels
    """
    C = _colors()
    blue, orange, violet = C["blue"], C["orange"], C["violet"]
    n_in, n_out = len(inputs), len(outputs)
    fig = plt.figure(figsize=figsize)
    bottom = 0.26 if n_in == 1 else 0.33
    gs = GridSpec(max(n_in, n_out, 1) * 2, 3, figure=fig, width_ratios=[1.1, 1.0, 1.0],
                  hspace=0.9, wspace=0.32, top=0.80, bottom=bottom, left=0.06, right=0.99)
    rows = max(n_in, n_out, 1) * 2

    # ---- left: what the branch sees -------------------------------------
    span = rows // n_in
    strips = []
    for i, inp in enumerate(inputs):
        ax = fig.add_subplot(gs[i * span:(i + 1) * span, 0])
        ax.plot(inp["x"], inp["y"], color=blue, lw=2, label=inp.get("label", "input"))
        ax.plot(inp["sx"], inp["sy"], "o", color=blue, ms=4.5, mec=fg(), mew=0.4,
                label=f"{len(inp['sx'])} sensors")
        ax.vlines(inp["sx"], np.minimum(0, inp["sy"]) if inp.get("stems", True) else inp["sy"],
                  inp["sy"], color=blue, alpha=0.3, lw=0.8)
        ax.axhline(0, color=muted(), lw=0.5)
        ax.set_xlabel(inp.get("xlabel", "$x$"))
        if inp.get("unit"):
            ax.set_ylabel(inp["unit"])
        ax.legend(fontsize=7, loc=inp.get("legend_loc", "best"))
        if i == 0:
            ax.set_title("what the branch sees", color=blue, fontsize=11, fontweight="bold")
        strips.append(f"{inp.get('symbol', 'a')} = {_vector_strip(inp['sy'])}  ({len(inp['sx'])} numbers)")

    # ---- middle: what the trunk sees ------------------------------------
    axq = fig.add_subplot(gs[:, 1])
    if query.get("kind", "1d") == "1d":
        if "modes" in query:
            for xm, pm, lab in query["modes"]:
                axq.plot(xm, pm, lw=1.5, label=lab)
            axq.set_title(query.get("modes_title", "stored modes read at $y$"), color=orange,
                          fontsize=11, fontweight="bold")
            axq.legend(fontsize=7)
        else:
            o = outputs[0]
            axq.plot(o["x"], o["y"], color=violet, lw=1.2, alpha=0.35)
            axq.set_title("what the trunk sees", color=orange, fontsize=11, fontweight="bold")
        q = query["q"]
        axq.axvline(q, color=orange, lw=1.6, ls=":")
        axq.plot([q], [0], "v", color=orange, ms=10, clip_on=False, zorder=5)
        axq.text(q, axq.get_ylim()[1], f" one query {query.get('qname', '$y$')} = {q:g}",
                 color=orange, fontsize=8, va="top", ha="left" if q < 0.6 * axq.get_xlim()[1] else "right")
        axq.axhline(0, color=muted(), lw=0.5)
        axq.set_xlabel(query.get("xlabel", "$y$"))
        if query.get("ylabel"):
            axq.set_ylabel(query["ylabel"])
    else:  # 2d space-time query
        field, xs, ts = query["field"], query["xs"], query["ts"]
        axq.pcolormesh(xs, ts, field, shading="auto", cmap="viridis", alpha=0.75)
        if "colloc" in query:
            cp = np.asarray(query["colloc"])
            axq.plot(cp[:, 0], cp[:, 1], ".", color=orange, ms=3, alpha=0.8, label="collocation points")
        axq.plot(xs, np.zeros_like(xs), color=blue, lw=3.5, label=query.get("input_label", "input lives on $t=0$"))
        qx, qt = query["q"]
        axq.plot([qx], [qt], "o", color=orange, ms=9, mec=fg(), mew=0.8, zorder=6, label=f"one query $(x,t)=({qx:g},{qt:g})$")
        axq.set_xlabel(query.get("xlabel", "$x$"))
        axq.set_ylabel(query.get("tlabel", "$t$"))
        axq.set_title("what the trunk sees: a point of the rectangle", color=orange, fontsize=11, fontweight="bold")
        axq.legend(fontsize=7, loc="upper right")

    # ---- right: what comes out ------------------------------------------
    span_o = rows // n_out
    for j, o in enumerate(outputs):
        ax = fig.add_subplot(gs[j * span_o:(j + 1) * span_o, 2])
        ax.plot(o["x"], o["y"], color=violet, lw=2, label=o.get("label", "output"))
        qx = o.get("qx", query["q"] if query.get("kind", "1d") == "1d" else query["q"][0])
        val = float(np.interp(qx, o["x"], o["y"]))
        ax.plot([qx], [val], "o", color=orange, ms=9, mec=fg(), mew=0.8, zorder=6)
        ax.annotate(f"{val:.3g}{o.get('unit_short', '')}", (qx, val), xytext=(10, -14), textcoords="offset points",
                    fontsize=9, color=fg(), arrowprops=dict(arrowstyle="-", color=fg(), lw=0.6))
        ax.axhline(0, color=muted(), lw=0.5)
        ax.set_xlabel(o.get("xlabel", query.get("xlabel", "$y$")))
        if o.get("ylabel"):
            ax.set_ylabel(o["ylabel"])
        ax.legend(fontsize=7, loc=o.get("legend_loc", "best"))
        if j == 0:
            ax.set_title("what comes out: one number per query", color=violet, fontsize=11, fontweight="bold")

    # ---- strips ----------------------------------------------------------
    y0 = 0.12 if n_in == 1 else 0.19
    for k, s in enumerate(strips):
        fig.text(0.06, y0 - 0.07 * k, s, fontsize=8.5, color=blue, family="monospace", va="top")
    fig.text(0.06, 0.015, dims, fontsize=9.5, color=fg(), va="bottom")
    if title:
        fig.suptitle(title, fontsize=12, y=0.97)
    return fig


def experiment_card(family, truth, compare, setup, figsize=(13.5, 4.6)):
    """Draw an experiment card: how the operator was trained and what is compared.

    family : dict
        x            common abscissa of the input functions
        curves       list of (y, split) with split in {'train', 'validation', 'test'}
        counts       dict split -> number of functions (None or missing = not used)
        how          one line: how the inputs are generated
        xlabel, ylabel, title_fn (name of the function, e.g. 'thrust histories')
    truth : dict
        x_in, y_in   one input function;  x_out, y_out  its solver output
        qx           query locations where the target is read
        solver       label of the solver, e.g. 'RK4, dt = 0.02 s, 1000 steps'
        in_label, out_label, xlabel_in, xlabel_out
    compare : dict
        x, truth, pred, qx, label_truth, label_pred, metric (text), xlabel
    setup : str    one line under the panels
    """
    C = _colors()
    blue, orange, violet, green = C["blue"], C["orange"], C["violet"], C["green"]
    split_color = {"train": blue, "validation": green, "test": orange}
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 1.05], hspace=0.75, wspace=0.3,
                  top=0.84, bottom=0.2, left=0.06, right=0.99)

    # 1. the input family and the split
    ax = fig.add_subplot(gs[:, 0])
    seen = set()
    for y, split in family["curves"]:
        lab = None
        if split not in seen:
            n = family.get("counts", {}).get(split)
            lab = f"{split}: {n} {family.get('title_fn', 'functions')}" if n else split
            seen.add(split)
        ax.plot(family["x"], y, color=split_color[split], lw=1.4, alpha=0.9 if split != "train" else 0.75, label=lab)
    for split in ("validation", "test"):
        if split not in seen and family.get("counts", {}).get(split):
            ax.plot([], [], color=split_color[split], label=f"{split}: {family['counts'][split]}")
    if not family.get("counts", {}).get("validation"):
        ax.plot([], [], " ", label="validation: none (settings fixed in advance)")
    ax.set_xlabel(family.get("xlabel", "$x$"))
    if family.get("ylabel"):
        ax.set_ylabel(family["ylabel"])
    ax.axhline(0, color=muted(), lw=0.5)
    ax.legend(fontsize=7, loc=family.get("legend_loc", "best"))
    ax.set_title("1. the input family, split by function", fontsize=11, fontweight="bold")
    ax.text(0.01, -0.3, family.get("how", ""), transform=ax.transAxes, fontsize=8, color=fg(), va="top", wrap=True)

    # 2. the truth: solver, read at the query points
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(truth["x_in"], truth["y_in"], color=blue, lw=1.8, label=truth.get("in_label", "input function"))
    ax1.legend(fontsize=7, loc="best")
    ax1.axhline(0, color=muted(), lw=0.5)
    ax1.set_title("2. the truth: solver, read at the queries", fontsize=11, fontweight="bold")
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(truth["x_out"], truth["y_out"], color=violet, lw=1.8, label=truth.get("out_label", "solver output"))
    qy = np.interp(truth["qx"], truth["x_out"], truth["y_out"])
    ax2.plot(truth["qx"], qy, "o", color=orange, ms=3.5, label=truth.get("q_label", f"targets at {len(truth['qx'])} queries"))
    ax2.set_xlabel(truth.get("xlabel_out", "")); ax2.legend(fontsize=7, loc="best")
    ax2.axhline(0, color=muted(), lw=0.5)
    fig.text(0.5, 0.515, f"$\\downarrow$ {truth.get('solver', 'solver')} $\\downarrow$", ha="center", fontsize=9, color=fg())

    # 3. what is compared
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.plot(compare["x"], compare["truth"], "--", color=fg(), lw=1.6, label=compare.get("label_truth", "truth (solver)"))
    ax3.plot(compare["x"], compare["pred"], color=violet, lw=2, label=compare.get("label_pred", "operator prediction"))
    if "qx" in compare:
        ax3.plot(compare["qx"], np.interp(compare["qx"], compare["x"], compare["truth"]), "o", color=orange, ms=3, alpha=0.7, label="compared at the queries")
    ax3.axhline(0, color=muted(), lw=0.5)
    ax3.set_xlabel(compare.get("xlabel", "")); ax3.legend(fontsize=7, loc=compare.get("legend_loc", "best"))
    ax3.set_title("3. what is compared, on a test function", fontsize=11, fontweight="bold")
    ax3.text(0.02, -0.3, compare.get("metric", ""), transform=ax3.transAxes, fontsize=8.5, color=fg(), va="top")

    fig.text(0.06, 0.015, setup, fontsize=9.5, color=fg(), va="bottom")
    return fig


def function_card(inputs, vectors, query, outputs, dims, qname="$y$", xlabel_in="$x$", xlabel_out="$y$",
                  ylabel_in=None, ylabel_out=None, title_in="1. the input functions", figsize=(14, 4.6)):
    """Several input functions, one colour each, and what the operator does with them.

    inputs  : list of dicts {x, y, sx, sy, label, color}  (one channel)
    vectors : list of (label, values, color)  printed as strips, one per input
    query   : dict {q, n_queries}  the one query marked for the trunk
    outputs : list of dicts {x, truth, pred, color}  (same order as inputs)
    dims    : one line under the panels
    """
    C = _colors()
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.15, 0.95, 0.7, 1.3], wspace=0.35, top=0.86, bottom=0.2, left=0.05, right=0.99)
    ax1 = fig.add_subplot(gs[0, 0])
    for inp in inputs:
        ax1.plot(inp["x"], inp["y"], color=inp["color"], lw=1.8, label=inp["label"])
        ax1.plot(inp["sx"], inp["sy"], "o", color=inp["color"], ms=3, mec=fg(), mew=0.3)
    ax1.axhline(0, color=muted(), lw=0.5); ax1.set_xlabel(xlabel_in)
    if ylabel_in: ax1.set_ylabel(ylabel_in)
    ax1.legend(fontsize=7, ncol=2); ax1.set_title(title_in, fontsize=10.5, fontweight="bold")
    ax1.text(0.02, 0.03, f"dots: the {len(inputs[0]['sx'])} sensors", transform=ax1.transAxes, fontsize=7.5)
    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off"); ax2.set_title("2. what the branch gets", fontsize=10.5, fontweight="bold", color=C["blue"])
    n = len(vectors)
    for row, (lab, vals, col) in enumerate(vectors):
        y0 = 0.92 - row * (0.9 / n)
        ax2.text(0.0, y0, f"{lab}:", color=col, fontsize=9, fontweight="bold", transform=ax2.transAxes)
        ax2.text(0.0, y0 - 0.08, _vector_strip(vals), color=col, fontsize=8, family="monospace", transform=ax2.transAxes)
        ax2.text(0.0, y0 - 0.15, f"{len(vals)} numbers", color=muted(), fontsize=7.5, transform=ax2.transAxes)
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_title("3. what the trunk gets", fontsize=10.5, fontweight="bold", color=C["orange"])
    qs = np.linspace(outputs[0]["x"].min(), outputs[0]["x"].max(), query.get("n_queries", 40))
    ax3.vlines(qs, 0, 0.35, color=muted(), lw=0.8); ax3.vlines([query["q"]], 0, 1, color=C["orange"], lw=2.5)
    ax3.text(query["q"], 1.02, f"one query\n{qname} = {query['q']:g}", color=C["orange"], fontsize=8.5, ha="center", va="bottom")
    ax3.text(0.5, 0.45, f"{query.get('n_queries', 40)} queries\n(ask all of them\nto draw the curve)", transform=ax3.transAxes, fontsize=8, ha="center", color=muted())
    ax3.set_ylim(0, 1.3); ax3.set_yticks([]); ax3.set_xlabel(xlabel_out); ax3.grid(False)
    ax4 = fig.add_subplot(gs[0, 3]); ax4.set_title("4. what comes out", fontsize=10.5, fontweight="bold", color=C["violet"])
    for o in outputs:
        ax4.plot(o["x"], o["truth"], "--", color=fg(), lw=1.0)
        ax4.plot(o["x"], o["pred"], color=o["color"], lw=2)
        v = float(np.interp(query["q"], o["x"], o["pred"]))
        ax4.plot([query["q"]], [v], "o", color=C["orange"], ms=7, mec=fg(), mew=0.6, zorder=5)
    ax4.plot([], [], "--", color=fg(), lw=1, label="solver truth"); ax4.plot([], [], color=muted(), lw=2, label="operator (one colour per input)")
    ax4.plot([], [], "o", color=C["orange"], ms=7, label=f"value at {qname} = {query['q']:g}")
    ax4.axhline(0, color=muted(), lw=0.5); ax4.set_xlabel(xlabel_out)
    if ylabel_out: ax4.set_ylabel(ylabel_out)
    ax4.legend(fontsize=7, loc="best")
    fig.text(0.05, 0.03, dims, fontsize=9, color=fg())
    return fig


def train_test_figure(x_in, train_inputs, train_targets, test_inputs, test_truth, test_pred, x_out,
                      n_train, n_test, err_mean, what="functions", xlabel_in="$x$", xlabel_out="$y$",
                      ylabel_in=None, ylabel_out=None, sensors=None, figsize=(12.5, 6.0)):
    """Left: training inputs and their solver targets. Right: test inputs, prediction vs truth."""
    C = _colors()
    fig, ax = plt.subplots(2, 2, figsize=figsize, gridspec_kw=dict(hspace=0.5, wspace=0.22))
    for y in train_inputs: ax[0, 0].plot(x_in, y, color=C["blue"], lw=1.1, alpha=0.75)
    if sensors is not None:
        ax[0, 0].plot(sensors, np.interp(sensors, x_in, train_inputs[0]), "o", color=C["blue"], ms=3, mec=fg(), mew=0.3)
    ax[0, 0].set_title(f"training {what}: {n_train} ({len(train_inputs)} shown)", fontsize=10, fontweight="bold", color=C["blue"])
    for y in test_inputs: ax[0, 1].plot(x_in, y, color=C["orange"], lw=1.3)
    ax[0, 1].set_title(f"test {what}: {n_test} the operator never saw ({len(test_inputs)} shown)", fontsize=10, fontweight="bold", color=C["orange"])
    for y in train_targets: ax[1, 0].plot(x_out, y, color=C["blue"], lw=1.1, alpha=0.75)
    ax[1, 0].set_title("their solver outputs: the targets the operator is fitted to", fontsize=10, fontweight="bold")
    for yt, yp in zip(test_truth, test_pred):
        ax[1, 1].plot(x_out, yt, "--", color=fg(), lw=1.0); ax[1, 1].plot(x_out, yp, color=C["orange"], lw=1.6)
    ax[1, 1].plot([], [], "--", color=fg(), lw=1, label="solver truth"); ax[1, 1].plot([], [], color=C["orange"], lw=1.6, label="operator prediction")
    ax[1, 1].legend(fontsize=8, loc="best")
    ax[1, 1].set_title(f"test outputs: prediction vs truth (mean error {err_mean:.1%} over the {n_test} test {what})", fontsize=10, fontweight="bold")
    for a in ax[0]: a.set_xlabel(xlabel_in); a.axhline(0, color=muted(), lw=0.5)
    for a in ax[1]: a.set_xlabel(xlabel_out); a.axhline(0, color=muted(), lw=0.5)
    if ylabel_in: ax[0, 0].set_ylabel(ylabel_in)
    if ylabel_out: ax[1, 0].set_ylabel(ylabel_out)
    plt.tight_layout()
    return fig
