"""Compose the README's still of the coverage grid from the stills and data.

    python grid_still.py [--site media/site] [--out media/site/coverage-grid.jpg]

A compact version of the page's figure for places that cannot run it - the
READMEs: each cell is its still, its status border and label, and its
seeds' training curves on the task's shared range, with the rows labelled
on the left so the cells stay near square. No notes or links; the image
links to the page, which has them. Drawn at twice its display size. Needs
Pillow; fonts default to DejaVu and can be named with --font / --bold /
--mono.
"""

import argparse
import json
import pathlib

from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
FONTS = pathlib.Path("/usr/share/fonts/truetype/dejavu")

INK = (31, 35, 40)
SOFT = (91, 101, 112)
GROUND = (255, 255, 255)
STATUS = {
    "learns": ((46, 125, 50), "learns", "learned the task"),
    "rising": ((141, 123, 0), "runs · rising", "runs, return still rising"),
    "flat": ((123, 135, 146), "runs · no learning", "runs, has not learned"),
    "runs": ((123, 135, 146), "runs end to end", None),
}

S = 2  # drawn at twice the display size
CELL = 150 * S  # the still's side
BORDER = 3 * S
CURVE = 30 * S
GAP = 10 * S
LEFT = 96 * S  # at least; widened to the longest row label
TOP = 26 * S
PAD = 8 * S


def curve_box(draw, box, curves, lo, hi, colour):
    x0, y0, x1, y1 = box
    seeds = [c for c in curves if len(c) > 1]
    if not seeds:
        return
    n = min(len(c) for c in seeds)

    def points(values):
        return [
            (
                x0 + (x1 - x0) * i / (n - 1),
                y1 - (y1 - y0) * ((v - lo) / (hi - lo) if hi > lo else 0.0),
            )
            for i, v in enumerate(values[:n])
        ]

    faint = tuple(int(c + (255 - c) * 0.6) for c in colour)
    for s in seeds:
        draw.line(points(s), fill=faint, width=1 * S)
    mean = [sum(s[i] for s in seeds) / len(seeds) for i in range(n)]
    draw.line(points(mean), fill=colour, width=2 * S, joint="curve")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=pathlib.Path, default=HERE / "media/site")
    parser.add_argument("--out", type=pathlib.Path, default=None)
    parser.add_argument("--font", default=str(FONTS / "DejaVuSans.ttf"))
    parser.add_argument("--bold", default=str(FONTS / "DejaVuSans-Bold.ttf"))
    parser.add_argument("--mono", default=str(FONTS / "DejaVuSansMono.ttf"))
    args = parser.parse_args()
    out = args.out or args.site / "coverage-grid.jpg"
    data = json.loads((args.site / "coverage.json").read_text())

    body = ImageFont.truetype(args.font, 11 * S)
    bold = ImageFont.truetype(args.bold, 11 * S)
    chip = ImageFont.truetype(args.bold, 9 * S)
    mono = ImageFont.truetype(args.mono, 10 * S)

    rows, cols = data["rows"], data["columns"]
    labels = [[p.strip() for p in row["label"].split("·")] for row in rows]
    left = max(
        LEFT,
        round(PAD + max(max(mono.getlength(p), bold.getlength(a)) for p, a in labels))
        + PAD,
    )
    cell_h = CELL + CURVE + 2 * BORDER
    width = left + len(cols) * (CELL + 2 * BORDER) + (len(cols) - 1) * GAP + PAD
    legend_y = TOP + len(rows) * cell_h + (len(rows) - 1) * GAP + 12 * S
    height = legend_y + 22 * S
    img = Image.new("RGB", (width, height), GROUND)
    draw = ImageDraw.Draw(img)

    def col_x(j):
        return left + j * (CELL + 2 * BORDER + GAP)

    def row_y(i):
        return TOP + i * (cell_h + GAP)

    for j, col in enumerate(cols):
        draw.text(
            (col_x(j) + BORDER, TOP - 6 * S),
            col["label"],
            font=bold,
            fill=SOFT,
            anchor="ls",
        )
    # Each row: the policy, and under it the algorithm that trains it.
    for i, (policy, algo) in enumerate(labels):
        mid = row_y(i) + cell_h / 2
        draw.text((PAD, mid - 3 * S), policy, font=mono, fill=INK, anchor="ls")
        draw.text((PAD, mid + 3 * S), algo, font=bold, fill=SOFT, anchor="lt")

    for c in data["cells"]:
        i = [r["id"] for r in rows].index(c["row"])
        j = [k["id"] for k in cols].index(c["column"])
        colour, label, _ = STATUS[c["status"]]
        x, y = col_x(j), row_y(i)
        draw.rounded_rectangle(
            (x, y, x + CELL + 2 * BORDER - 1, y + cell_h - 1), radius=5 * S, fill=colour
        )
        draw.rectangle(
            (
                x + BORDER,
                y + BORDER + CELL,
                x + BORDER + CELL - 1,
                y + cell_h - BORDER - 1,
            ),
            fill=GROUND,
        )
        still = (
            Image.open(args.site / f"{c['id']}.jpg")
            .convert("RGB")
            .resize((CELL, CELL), Image.LANCZOS)
        )
        img.paste(still, (x + BORDER, y + BORDER))
        tw = draw.textlength(label, font=chip)
        cx, cy = x + BORDER + 5 * S, y + BORDER + 5 * S
        draw.rounded_rectangle(
            (cx, cy, cx + tw + 10 * S, cy + 15 * S), radius=3 * S, fill=colour
        )
        draw.text(
            (cx + 5 * S, cy + 7.5 * S), label, font=chip, fill=GROUND, anchor="lm"
        )
        lo, hi = data["ranges"][c["column"]]
        curve_box(
            draw,
            (
                x + BORDER + 6 * S,
                y + BORDER + CELL + 5 * S,
                x + BORDER + CELL - 6 * S,
                y + cell_h - BORDER - 5 * S,
            ),
            c["curves"],
            lo,
            hi,
            colour,
        )

    lx = left
    for status in ("learns", "rising", "flat"):
        colour, _, text = STATUS[status]
        draw.rounded_rectangle(
            (lx, legend_y, lx + 11 * S, legend_y + 11 * S),
            radius=2 * S,
            outline=colour,
            width=3 * S,
        )
        draw.text(
            (lx + 17 * S, legend_y + 5.5 * S), text, font=body, fill=SOFT, anchor="lm"
        )
        lx += 17 * S + draw.textlength(text, font=body) + 18 * S

    img.save(out, quality=86)
    print(f"wrote {out}: {width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
