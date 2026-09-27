"""Derive E26's harnesses from E14's by exact substitution, and fail on a miss."""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def derive(src: str, dst: str, subs: list[tuple[str, str]]) -> None:
    text = (HERE / src).read_text(encoding="utf-8")
    for old, new in subs:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"{src}: expected exactly one of {old!r}, found {n}")
        text = text.replace(old, new)
    (HERE / dst).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {dst}")


SERVER_CD = 'cd "$R/plugrl-server" && exec setsid env \\\n'
SERVER_CD_E26 = (
    'cd "$R/plugrl-server-e26" && exec setsid env \\\n'
    '    PYTHONPATH="$R/plugrl-server-e26/src" \\\n'
)

derive(
    "e14_train.sh",
    "train.sh",
    [
        (
            "#   bash e14_feasibility.sh CELL ITERATIONS PORT [MASTER_DEVICE]",
            "#   bash e26/train.sh CELL ITERATIONS PORT [MASTER_DEVICE]\n"
            "#\n"
            "# E26: E14's training harness, unchanged but for three things - the\n"
            "# server runs $R/plugrl-server-e26 (main plus #60, deployed LF) through\n"
            "# PYTHONPATH, output goes under e26/, and FREEZE, when set, is passed\n"
            "# as --policy.freeze-expert-params.",
        ),
        ("OUT=$R/e14/$CELL", "OUT=$R/e26/$CELL"),
        (SERVER_CD, SERVER_CD_E26),
        (
            '[ -n "${DRIFT:-}" ] && [ "${DRIFT:-0}" != 0 ] \\\n'
            '  && OPT_FLAGS+=(--algo.max-policy-drift "$DRIFT")\n',
            '[ -n "${DRIFT:-}" ] && [ "${DRIFT:-0}" != 0 ] \\\n'
            '  && OPT_FLAGS+=(--algo.max-policy-drift "$DRIFT")\n'
            '[ -n "${FREEZE:-}" ] \\\n'
            '  && OPT_FLAGS+=(--policy.freeze-expert-params "$FREEZE")\n',
        ),
        ('log "E14_FEASIBILITY_DONE $CELL"', 'log "E26_TRAIN_DONE $CELL"'),
    ],
)

derive(
    "e14_eval.sh",
    "eval.sh",
    [
        (
            "# E14 evaluation harness. Derived by sed from e11/e11_stageC_eval.sh; only the",
            "# E26 evaluation harness: E14's, with the server on $R/plugrl-server-e26\n"
            "# through PYTHONPATH and output under e26/. E14's own header follows.\n"
            "#\n"
            "# E14 evaluation harness. Derived by sed from e11/e11_stageC_eval.sh; only the",
        ),
        ("E=$R/e14\n", "E=$R/e26\n"),
        ("RES=${E14_RES:-$E/results}", "RES=${E26_RES:-$E/results}"),
        (SERVER_CD, SERVER_CD_E26),
    ],
)
