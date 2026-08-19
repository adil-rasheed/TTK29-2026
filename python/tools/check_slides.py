#!/usr/bin/env python3
"""Find slide content that runs off the bottom of the frame.

Why this exists: ``latexmk`` reporting zero overfull boxes does NOT mean the
deck is clean. The NTNU theme's ``\\textheight`` extends below the visible
slide, so a frame can overflow the *slide* without overflowing the *text
block* -- LaTeX stays silent, beamer's ``[shrink]`` never fires, and the
rendered page quietly loses its last lines. That failure is invisible in the
build log and obvious to anyone in the lecture theatre.

Detection is geometric. The theme's ``footline`` template (see
``latex/common/theme/beamerouterthementnu.sty``) paints an opaque
``beamercolorbox`` of height ``1.5em + 0.4em`` across the full page width
*after* the frame body is typeset. Anything the body left in that band is
covered up. So the usable bottom of a slide is the band's top edge, not the
page edge -- and not the page-number glyphs, which sit *inside* the band and
would put the baseline about 10 pt too low.

    python tools/check_slides.py            # every built deck
    python tools/check_slides.py 16-conv    # substring filter
    python tools/check_slides.py --dpi 110 --save /tmp   # dump the offenders

Requires PyMuPDF. Exits non-zero if any page overflows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    sys.exit("check_slides needs PyMuPDF:  pip install pymupdf")

ROOT = Path(__file__).resolve().parents[2]
MODULES = ROOT / "latex" / "modules"

# Text may descend this far into the band before we call it clipped: enough
# to absorb descenders and rounding, far less than a line of body text.
TOLERANCE_PT = 2.5


def text_bottom(page) -> float:
    """Lowest point of any non-empty text block on the page."""
    return max(
        [b[3] for b in page.get_text("blocks") if b[4].strip()] or [0.0]
    )


def band_top(doc) -> float:
    """Top edge of the opaque footer band -- the true bottom of usable space.

    Measured from the page rather than hard-coded, so a theme change cannot
    silently invalidate the check: the band is the full-width filled rectangle
    nearest the bottom of an interior page.
    """
    page = doc[min(1, doc.page_count - 1)]
    H, W = page.rect.height, page.rect.width
    tops = [
        d["rect"].y0
        for d in page.get_drawings()
        if d["rect"].width > 0.9 * W and d["rect"].y1 > H - 1.0
    ]
    if tops:
        return max(tops)
    return H - 11.4  # 1.5em + 0.4em at 11pt, the template's declared height


def check(pdf: Path) -> list[tuple[int, float]]:
    doc = fitz.open(pdf)
    base = band_top(doc)
    bad = []
    for i, page in enumerate(doc):
        if i == 0 or i == doc.page_count - 1:
            continue  # title and closing slides are decorative by design
        # Work per LINE, not per block, and ignore lines that sit wholly
        # inside the footer band -- those are the page number and module
        # name, which the template draws there on purpose.
        #
        # Two earlier rules both failed, in opposite directions, and both at
        # block granularity:
        #   * "ignore blocks ending more than 9 pt below the band top"
        #     exempted the WORST overflows, since a caption running far off
        #     the page looks exactly like a page number. False negatives that
        #     grew with the severity of the fault.
        #   * "ignore blocks starting inside the band" fails because PyMuPDF
        #     merges a body block and the footer's page number into a single
        #     block when they are vertically adjacent, so the merged block
        #     starts above the band and ends inside it. False positives.
        # A line is small enough that neither confusion arises.
        bottoms = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if not any(s["text"].strip() for s in line["spans"]):
                    continue
                y0, y1 = line["bbox"][1], line["bbox"][3]
                if y0 >= base - 1.0:
                    continue          # lives in the band: footer furniture
                bottoms.append(y1)
        over = (max(bottoms) if bottoms else 0.0) - base
        if over > TOLERANCE_PT:
            bad.append((i + 1, round(over, 1)))
    doc.close()
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("filter", nargs="?", default="", help="module name substring")
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--save", default="", help="write offending pages as PNGs here")
    args = ap.parse_args()

    pdfs = sorted(MODULES.glob("*/slides/build/slides.pdf"))
    pdfs = [p for p in pdfs if args.filter in str(p)]
    if not pdfs:
        print("no built decks found (build the slides first)")
        return 1

    total = 0
    for pdf in pdfs:
        module = pdf.parents[2].name
        bad = check(pdf)
        total += len(bad)
        if bad:
            pages = ", ".join(f"p{n} (+{o}pt)" for n, o in bad)
            print(f"CLIPPED  {module}: {pages}")
            if args.save:
                doc = fitz.open(pdf)
                for n, _ in bad:
                    out = Path(args.save) / f"{module}_p{n}.png"
                    doc[n - 1].get_pixmap(dpi=args.dpi).save(out)
                doc.close()
        else:
            print(f"clean    {module}")

    print(
        f"\n{total} clipped page(s) across {len(pdfs)} deck(s)."
        if total
        else f"\nall {len(pdfs)} deck(s) clean."
    )
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
