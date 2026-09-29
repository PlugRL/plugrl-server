"""Read E45's verdicts from results/footprint.tsv, in PROTOCOL.md's order.

python summarise.py [results-dir]
"""

import csv
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
R = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "results"
NO_TORCH = ("plugrl-env-client", "plugrl-minimal", "dm-env-rpc", "openpi-client")
TORCH = ("rllib-rllink", "lerobot-hilserl")


def main() -> int:
    rows = {
        r["name"]: r
        for r in csv.DictReader(
            open(R / "footprint.tsv", encoding="utf-8"), delimiter="\t"
        )
    }
    mb = {k: int(r["bytes"]) / 1e6 for k, r in rows.items()}

    print("P4  every install works and its entry module imports")
    p4 = True
    for k, r in rows.items():
        ok = r["install_rc"] == "0" and r["entry_import"] == "ok"
        p4 &= ok
        print(f"    {k:18s} install rc {r['install_rc']}  import {r['entry_import']}")
    print(f"    {'HOLDS' if p4 and len(rows) == 6 else 'FALSIFIED'}")

    print("\nP1  torch only where the environment side runs the policy")
    p1 = True
    for k in NO_TORCH:
        ok = rows[k]["torch"] == "none"
        p1 &= ok
        print(
            f"    {k:18s} torch {rows[k]['torch']:10s} {'as predicted' if ok else 'WRONG'}"
        )
    for k in TORCH:
        ok = rows[k]["torch"] != "none" and int(rows[k]["nvidia_packages"]) >= 1
        p1 &= ok
        print(
            f"    {k:18s} torch {rows[k]['torch']:10s} nvidia {rows[k]['nvidia_packages']:>2s}  {'as predicted' if ok else 'WRONG'}"
        )
    print(f"    {'HOLDS' if p1 else 'FALSIFIED'}")

    print("\nP2  plugrl-env-client at least 10x smaller than each torch-bearing side")
    p2 = True
    for k in TORCH:
        ratio = mb[k] / mb["plugrl-env-client"]
        p2 &= ratio >= 10
        print(
            f"    {k:18s} {mb[k]:9.1f} MB / {mb['plugrl-env-client']:.1f} MB = {ratio:.1f}x"
        )
    print(f"    {'HOLDS' if p2 else 'FALSIFIED'}")

    print("\nP3  dm-env-rpc and openpi-client within 3x of plugrl-env-client")
    p3 = True
    for k in ("dm-env-rpc", "openpi-client"):
        ratio = mb[k] / mb["plugrl-env-client"]
        p3 &= 1 / 3 <= ratio <= 3
        print(
            f"    {k:18s} {mb[k]:9.1f} MB / {mb['plugrl-env-client']:.1f} MB = {ratio:.2f}x"
        )
    print(f"    {'HOLDS' if p3 else 'FALSIFIED'}")

    print("\nreported: every system")
    print(
        f"    {'system':18s} {'packages':>8s} {'MB':>9s} {'nvidia':>6s} {'seconds':>8s}"
    )
    for k, r in rows.items():
        print(
            f"    {k:18s} {r['packages']:>8s} {mb[k]:9.1f} {r['nvidia_packages']:>6s} {r['seconds']:>8s}"
        )
    print("    plugrl-cpp                0       0.1      0        -   (E44's binary)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
