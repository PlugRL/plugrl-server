"""Derive E32's harnesses from E14's by exact substitution, and fail on a miss."""

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
SERVER_CD_E32 = (
    'cd "$R/plugrl-server-e32" && exec setsid env \\\n'
    '    PYTHONPATH="$R/plugrl-server-e32/src" \\\n'
)

derive(
    "e14_train.sh",
    "train.sh",
    [
        (
            "#   bash e14_feasibility.sh CELL ITERATIONS PORT [MASTER_DEVICE]",
            "#   bash e32/train.sh CELL ITERATIONS PORT [MASTER_DEVICE]\n"
            "#\n"
            "# E32: E14's training harness, unchanged but for three things - the\n"
            "# server runs $R/plugrl-server-e32 (main plus #74, deployed LF) through\n"
            "# PYTHONPATH, output goes under e32/, and ARM_FLAGS, when set, is split\n"
            "# on spaces and passed to the server after every other flag.",
        ),
        ("OUT=$R/e14/$CELL", "OUT=$R/e32/$CELL"),
        (SERVER_CD, SERVER_CD_E32),
        (
            '[ -n "${DRIFT:-}" ] && [ "${DRIFT:-0}" != 0 ] \\\n'
            '  && OPT_FLAGS+=(--algo.max-policy-drift "$DRIFT")\n',
            '[ -n "${DRIFT:-}" ] && [ "${DRIFT:-0}" != 0 ] \\\n'
            '  && OPT_FLAGS+=(--algo.max-policy-drift "$DRIFT")\n'
            'if [ -n "${ARM_FLAGS:-}" ]; then\n'
            '  read -r -a _ARM <<< "$ARM_FLAGS"\n'
            '  OPT_FLAGS+=("${_ARM[@]}")\n'
            "fi\n"
            'log "arm flags: ${ARM_FLAGS:-none}"\n',
        ),
        ('log "E14_FEASIBILITY_DONE $CELL"', 'log "E32_TRAIN_DONE $CELL"'),
    ],
)

derive(
    "e14_eval.sh",
    "eval.sh",
    [
        (
            "# E14 evaluation harness. Derived by sed from e11/e11_stageC_eval.sh; only the",
            "# E32 evaluation harness: E14's, with the server on $R/plugrl-server-e32\n"
            "# through PYTHONPATH, output under e32/, and the source manifest taken\n"
            "# through the same PYTHONPATH - E25's and E26's hashed the older copy the\n"
            "# editable install points at. E14's own header follows.\n"
            "#\n"
            "# E14 evaluation harness. Derived by sed from e11/e11_stageC_eval.sh; only the",
        ),
        ("E=$R/e14\n", "E=$R/e32\n"),
        ("RES=${E14_RES:-$E/results}", "RES=${E32_RES:-$E/results}"),
        (
            '  source_manifest "$SPY" plugrl_server\n',
            '  PYTHONPATH="$R/plugrl-server-e32/src" source_manifest "$SPY" plugrl_server\n',
        ),
        (SERVER_CD, SERVER_CD_E32),
    ],
)
