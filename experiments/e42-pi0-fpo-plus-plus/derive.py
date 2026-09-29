"""Derive E42's harnesses from E36's by exact substitution, and fail on a miss.

E36's train.sh kept E32's 12-hour client timeout, which ten FPO iterations of
about 80 minutes outran: the tenth learn step was killed mid-flight. E42's
is 30 hours. Everything else changes only where the output, the code
directory and the log markers live.
"""

import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def derive(src: str, dst: str, subs: list[tuple[str, str, int]]) -> None:
    text = (HERE / src).read_text(encoding="utf-8")
    for old, new, count in subs:
        n = text.count(old)
        if n != count:
            raise SystemExit(f"{src}: expected {count} of {old!r}, found {n}")
        text = text.replace(old, new)
    (HERE / dst).write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {dst}")


derive(
    "e36_train.sh",
    "train.sh",
    [
        (
            "#   bash e36/train.sh CELL ITERATIONS PORT [MASTER_DEVICE]\n",
            "#   bash e42/train.sh CELL ITERATIONS PORT [MASTER_DEVICE]\n"
            "#\n"
            "# E42: E36's training harness with its output under e42/, the server\n"
            "# on $R/plugrl-server-e42 (#91), and a 30-hour client timeout where\n"
            "# E36's 12 hours cut its tenth iteration. E36's header follows.\n"
            "#\n"
            "#   bash e36/train.sh CELL ITERATIONS PORT [MASTER_DEVICE]\n",
            1,
        ),
        ("OUT=$R/e36/$CELL", "OUT=$R/e42/$CELL", 1),
        (
            '  cd "$R/plugrl-server-e32" && exec setsid env \\',
            '  cd "$R/plugrl-server-e42" && exec setsid env \\',
            1,
        ),
        (
            '    PYTHONPATH="$R/plugrl-server-e32/src" \\',
            '    PYTHONPATH="$R/plugrl-server-e42/src" \\',
            1,
        ),
        ("    timeout 43200 ", "    timeout 108000 ", 1),
        ('log "E36_TRAIN_DONE $CELL"', 'log "E42_TRAIN_DONE $CELL"', 1),
    ],
)

derive(
    "e36_eval.sh",
    "eval.sh",
    [
        (
            "# E36 evaluation harness: E32's with output under e36/. E32's header:",
            "# E42 evaluation harness: E36's with output under e42/ and the server on\n"
            "# $R/plugrl-server-e42. E36's header:\n"
            "#\n"
            "# E36 evaluation harness: E32's with output under e36/. E32's header:",
            1,
        ),
        ("E=$R/e36\n", "E=$R/e42\n", 1),
        ("RES=${E36_RES:-$E/results}", "RES=${E42_RES:-$E/results}", 1),
        (
            'PYTHONPATH="$R/plugrl-server-e32/src"',
            'PYTHONPATH="$R/plugrl-server-e42/src"',
            2,
        ),
        (
            '  cd "$R/plugrl-server-e32" && exec setsid env \\',
            '  cd "$R/plugrl-server-e42" && exec setsid env \\',
            1,
        ),
    ],
)
