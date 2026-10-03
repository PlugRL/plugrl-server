"""E57, exploratory (not in PROTOCOL.md): are the runs both curve checks miss
shifted as a group, though no single one is harmed?

    python explore.py [results]

For each env and dose, the clean returns of boundary-fault runs that are
silent on the curve, in units of the fault-free band (none, seeds 10-19):
their mean z, and a two-sided Mann-Whitney U test against the 13 fault-free
runs (seeds 0-2 and 10-19). Also the clean-return gap between each such run
and the fault-free run at its own seed. Writes results/explore.txt.
"""

import csv
import pathlib
import statistics as st
import sys

from scipy.stats import mannwhitneyu

HERE = pathlib.Path(__file__).resolve().parent


def main() -> int:
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    rows = list(csv.DictReader(open(root / "clean.csv")))
    for r in rows:
        r["seed"], r["clean"] = int(r["seed"]), float(r["clean"])
    out = ["exploratory: boundary-fault runs silent on the curve, against fault-free runs",
           "env          dose    n   mean z   median z   U-test p   twin gap median (clean - none at same seed)"]
    for env in sorted({r["env"] for r in rows}):
        band = [r["clean"] for r in rows if r["env"] == env and r["fault"] == "none" and r["seed"] >= 10]
        mu, sd = st.mean(band), st.stdev(band)
        free = [r["clean"] for r in rows if r["env"] == env and r["fault"] == "none"]
        twin = {r["seed"]: r["clean"] for r in rows if r["env"] == env and r["fault"] == "none"}
        for dose in ("one", "0.001", "0.01", "0.1", "1.0"):
            s = [r for r in rows if r["env"] == env and r["dose"] == dose and r["kind"] == "boundary"
                 and r["seed"] < 10 and r["silent_curve"] == "1"]
            if not s:
                continue
            z = [(r["clean"] - mu) / sd for r in s]
            p = mannwhitneyu([r["clean"] for r in s], free, alternative="two-sided").pvalue
            gap = st.median(r["clean"] - twin[r["seed"]] for r in s)
            out.append(f"{env:12} {dose:6} {len(s):3}   {st.mean(z):6.2f}   {st.median(z):8.2f}   {p:8.3f}   {gap:9.1f}")
    text = "\n".join(out)
    (root / "explore.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
