"""Derive E25's evaluation harness from E14's by exact substitution, and fail on a miss."""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent

SERVER_CD = 'cd "$R/plugrl-server" && exec setsid env \\\n'
SERVER_CD_E25 = (
    'cd "$R/plugrl-server-main" && exec setsid env \\\n'
    '    PYTHONPATH="$R/plugrl-server-main/src" \\\n'
)

SUBS = [
    (
        "# E14 evaluation harness. Derived by sed from e11/e11_stageC_eval.sh; only the",
        "# E25 evaluation harness: E14's, with the server on $R/plugrl-server-main\n"
        "# through PYTHONPATH and output under e25/. E14's own header follows.\n"
        "#\n"
        "# E14 evaluation harness. Derived by sed from e11/e11_stageC_eval.sh; only the",
    ),
    ("E=$R/e14\n", "E=$R/e25\n"),
    ("RES=${E14_RES:-$E/results}", "RES=${E25_RES:-$E/results}"),
    (SERVER_CD, SERVER_CD_E25),
]

text = (HERE / "e14_eval.sh").read_text(encoding="utf-8")
for old, new in SUBS:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"e14_eval.sh: expected exactly one of {old!r}, found {n}")
    text = text.replace(old, new)
(HERE / "eval.sh").write_text(text, encoding="utf-8", newline="\n")
print("wrote eval.sh")
