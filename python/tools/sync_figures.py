#!/usr/bin/env python3
"""Figure bridge: python/<module>/figures  ->  latex/modules/<module>/figures.

The notebooks are the single source of truth for figures.  This script copies
their output to the LaTeX side and audits the result:

  * STALE    latex copy differs from the python original (fixed by syncing)
  * ORPHAN   latex figure no notebook produces (warning; error with --strict)
  * MISSING  \\includegraphics target that exists on neither side (always error)

Usage
-----
  python tools/sync_figures.py           # sync, then report
  python tools/sync_figures.py --check   # report only, no writes; nonzero exit
                                         # on STALE or MISSING
  python tools/sync_figures.py --check --strict   # orphans are errors too

Run from anywhere; paths are resolved relative to this file.
"""

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
PYTHON = ROOT / "python"
LATEX_MODULES = ROOT / "latex" / "modules"
THEME_DIR = ROOT / "latex" / "common" / "theme"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".pdf")
INCLUDE_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_pairs():
    """Yield (name, python_figdir_or_None, latex_figdir) per latex module."""
    for latex_mod in sorted(LATEX_MODULES.iterdir()):
        if not latex_mod.is_dir():
            continue
        py_fig = PYTHON / latex_mod.name / "figures"
        yield latex_mod.name, (py_fig if py_fig.is_dir() else None), latex_mod / "figures"


def tex_files_for(module_dir: Path):
    yield from module_dir.rglob("*.tex")


def strip_comments(line: str) -> str:
    # drop everything after an unescaped %
    out = []
    prev = ""
    for ch in line:
        if ch == "%" and prev != "\\":
            break
        out.append(ch)
        prev = ch
    return "".join(out)


def referenced_figures(module_dir: Path):
    """All \\includegraphics basenames referenced by a module's .tex files.

    A macro DEFINITION containing \\includegraphics{#2} is not a reference to
    a file called ``#2``; skip anything holding a macro parameter, or the
    audit reports a phantom missing figure for every such helper.
    """
    refs = set()
    for tex in tex_files_for(module_dir):
        if "build" in tex.parts:
            continue
        for line in tex.read_text(errors="replace").splitlines():
            for m in INCLUDE_RE.finditer(strip_comments(line)):
                name = m.group(1)
                if "#" in name:
                    continue          # macro parameter, not a filename
                refs.add(Path(name).name)
    return refs


def resolve(name: str, search_dirs):
    """Find an includegraphics target, with or without extension."""
    candidates = [name] if Path(name).suffix else [name + e for e in IMG_EXTS]
    for d in search_dirs:
        for c in candidates:
            if (d / c).is_file():
                return d / c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report only, no copying")
    ap.add_argument("--strict", action="store_true", help="orphans are errors")
    args = ap.parse_args()

    synced, stale, orphans, missing = [], [], [], []

    for name, py_fig, tex_fig in module_pairs():
        py_pngs = {p.name: p for p in py_fig.glob("*.png")} if py_fig else {}
        tex_pngs = {p.name: p for p in tex_fig.glob("*.png")} if tex_fig.is_dir() else {}

        # ---- sync python -> latex --------------------------------------
        for fname, src in sorted(py_pngs.items()):
            dst = tex_fig / fname
            if not dst.is_file() or sha256(src) != sha256(dst):
                stale.append(f"{name}/{fname}")
                if not args.check:
                    tex_fig.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    synced.append(f"{name}/{fname}")

        # ---- latex figures nothing produces ----------------------------
        for fname in sorted(tex_pngs):
            if py_pngs and fname not in py_pngs:
                orphans.append(f"{name}/{fname}")
            elif not py_pngs:
                orphans.append(f"{name}/{fname} (module has no python figures)")

        # ---- missing includegraphics targets ---------------------------
        # cross-module reuse is legitimate (e.g. the intro deck shows other
        # modules' flagship figures), so search every module's figures dir
        module_dir = LATEX_MODULES / name
        all_figdirs = [d / "figures" for d in sorted(LATEX_MODULES.iterdir())
                       if (d / "figures").is_dir()]
        search = [tex_fig, THEME_DIR, module_dir] + all_figdirs
        for ref in sorted(referenced_figures(module_dir)):
            if resolve(ref, search) is None:
                # after a sync the file may just have arrived
                if not (tex_fig / ref).is_file():
                    missing.append(f"{name}: {ref}")

    # ---- report --------------------------------------------------------
    if synced:
        print(f"SYNCED  ({len(synced)}):")
        for s in synced:
            print(f"  {s}")
    if stale and args.check:
        print(f"STALE   ({len(stale)}) — latex copy differs from python original:")
        for s in stale:
            print(f"  {s}")
    if orphans:
        print(f"ORPHAN  ({len(orphans)}) — no notebook produces these:")
        for s in orphans:
            print(f"  {s}")
    if missing:
        print(f"MISSING ({len(missing)}) — referenced but found nowhere:")
        for s in missing:
            print(f"  {s}")
    if not any([synced, stale if args.check else [], orphans, missing]):
        print("clean: latex figures match python originals; all references resolve")

    fail = bool(missing) or (args.check and bool(stale)) or (args.strict and bool(orphans))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
