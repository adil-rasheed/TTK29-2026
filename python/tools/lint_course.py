#!/usr/bin/env python3
"""Course consistency linter.

Enforces the conventions stated in courseslides.sty / coursenotes.sty:

  notation   scalar = italic non-bold  x
             vector = bold ITALIC      \\boldsymbol{x}, \\boldsymbol{\\theta}
             matrix = bold UPRIGHT     \\mathbf{A}, \\boldsymbol{\\Sigma}
             written out in full -- so:
               N1  no \\bm{...}
               N2  no \\vec{...}
               N3  no \\newcommand notation shortcuts (\\newcommand{\\vx}{...} etc.)
               N4  no \\mathbf{x} on a lowercase letter -- \\mathbf is bold
                   UPRIGHT, which is the matrix style. Vectors are bold italic,
                   so they need \\boldsymbol. \\mathbf{0} and \\mathbf{1} are
                   exempt: the zero and ones vectors stay upright by decision.
  colors     C1  no \\definecolor / \\colorlet outside latex/common/
  figures    F1  slides must include the *_dark.png variant of dual figures
             F2  chapters/notes/theory must include the light (plain) variant

F1/F2 only fire when BOTH variants of a figure exist, so the rule activates
module-by-module as the dual pipeline (Phase 1) lands.

Usage:  python tools/lint_course.py [--baseline]
        --baseline prints counts only (no per-line detail), for progress tracking.
Exit:   nonzero when any violation is found.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / "latex"
COMMON = LATEX / "common"

INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")

CHECKS = {
    "N1-bm": re.compile(r"\\bm\{"),
    "N2-vec": re.compile(r"\\vec\{"),
    "N3-shortcut": re.compile(
        r"\\(?:new|provide|renew)command\{\\(?:m[A-Z]|v[a-z]+|vone)\}"),
    # Digits are exempt, so \mathbf{0} and \mathbf{1} pass; so is anything
    # multi-token, which is table emphasis like $\mathbf{4058}$, not notation.
    "N4-upright-vector": re.compile(r"\\mathbf\s*\{\s*[a-z](?:_\{?[a-zA-Z0-9]\}?)?\s*\}"),
    "C1-localcolor": re.compile(r"\\(?:definecolor|colorlet|providecolor)\b"),
}


def strip_comments(line: str) -> str:
    out, prev = [], ""
    for ch in line:
        if ch == "%" and prev != "\\":
            break
        out.append(ch)
        prev = ch
    return "".join(out)


def tex_sources():
    for tex in LATEX.rglob("*.tex"):
        if "build" in tex.parts:
            continue
        yield tex


def lint_notation_and_colors(violations):
    for tex in tex_sources():
        in_common = COMMON in tex.parents
        rel = tex.relative_to(ROOT)
        for lineno, raw in enumerate(tex.read_text(errors="replace").splitlines(), 1):
            line = strip_comments(raw)
            if not line.strip():
                continue
            for key, rx in CHECKS.items():
                if key == "C1-localcolor" and in_common:
                    continue        # palette definitions belong in common/
                if rx.search(line):
                    violations.setdefault(key, []).append(f"{rel}:{lineno}: {line.strip()[:90]}")


def lint_notebook_notation(violations):
    """Figure labels are notation too, and the easiest place for drift to hide.

    Only the N-rules apply: a notebook has no colours or figure variants to get
    wrong. A .ipynb stores TeX inside JSON as ``\\\\mathbf``, which the same
    patterns match from the second backslash on.
    """
    py = ROOT / "python"
    for src in sorted(list(py.rglob("*.py")) + list(py.rglob("*.ipynb"))):
        if ".ipynb_checkpoints" in src.parts or "tools" in src.relative_to(py).parts:
            continue
        rel = src.relative_to(ROOT)
        for lineno, line in enumerate(src.read_text(errors="replace").splitlines(), 1):
            for key, rx in CHECKS.items():
                if not key.startswith("N"):
                    continue
                if rx.search(line):
                    violations.setdefault(key, []).append(
                        f"{rel}:{lineno}: {line.strip()[:90]}")


def lint_figure_variants(violations):
    """Slides use *_dark.png, print media use the light twin — where both exist."""
    modules = LATEX / "modules"
    for mod in sorted(modules.iterdir()) if modules.is_dir() else []:
        figdir = mod / "figures"
        if not figdir.is_dir():
            continue
        pngs = {p.name for p in figdir.glob("*.png")}
        duals = {n for n in pngs
                 if n.endswith("_dark.png") and n[:-9] + ".png" in pngs}
        light_of = {d: d[:-9] + ".png" for d in duals}
        dark_of = {v: k for k, v in light_of.items()}

        for tex in mod.rglob("*.tex"):
            if "build" in tex.parts:
                continue
            is_slides = "slides" in tex.parts
            rel = tex.relative_to(ROOT)
            for lineno, raw in enumerate(tex.read_text(errors="replace").splitlines(), 1):
                for m in INCLUDE_RE.finditer(strip_comments(raw)):
                    name = Path(m.group(1)).name
                    if not Path(name).suffix:
                        name += ".png"
                    if is_slides and name in dark_of:
                        violations.setdefault("F1-slides-light", []).append(
                            f"{rel}:{lineno}: {name} (use {dark_of[name]})")
                    elif not is_slides and name in light_of:
                        violations.setdefault("F2-print-dark", []).append(
                            f"{rel}:{lineno}: {name} (use {light_of[name]})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", action="store_true", help="counts only")
    args = ap.parse_args()

    violations = {}
    lint_notation_and_colors(violations)
    lint_notebook_notation(violations)
    lint_figure_variants(violations)

    if not violations:
        print("lint clean")
        return 0

    total = 0
    for key in sorted(violations):
        items = violations[key]
        total += len(items)
        print(f"{key}: {len(items)}")
        if not args.baseline:
            for it in items[:200]:
                print(f"  {it}")
            if len(items) > 200:
                print(f"  ... and {len(items) - 200} more")
    print(f"TOTAL: {total} violations")
    return 1


if __name__ == "__main__":
    sys.exit(main())
