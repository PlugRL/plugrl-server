"""Derive E36's harnesses from E32's and E25's by exact substitution, and fail on a miss."""

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


derive(
    "e32_train.sh",
    "train.sh",
    [
        (
            "#   bash e32/train.sh CELL ITERATIONS PORT [MASTER_DEVICE]",
            "#   bash e36/train.sh CELL ITERATIONS PORT [MASTER_DEVICE]\n"
            "#\n"
            "# E36: E32's training harness with its output under e36/ and the\n"
            "# server's seed taken from SEED (default 7, E14's). E32's header follows.\n"
            "#\n"
            "#   bash e32/train.sh CELL ITERATIONS PORT [MASTER_DEVICE]",
        ),
        ("OUT=$R/e32/$CELL", "OUT=$R/e36/$CELL"),
        ("      --seed 7 \\\n", "      --seed ${SEED:-7} \\\n"),
        ('log "E32_TRAIN_DONE $CELL"', 'log "E36_TRAIN_DONE $CELL"'),
    ],
)

derive(
    "e32_eval.sh",
    "eval.sh",
    [
        (
            "# E32 evaluation harness: E14's, with the server on $R/plugrl-server-e32",
            "# E36 evaluation harness: E32's with output under e36/. E32's header:\n"
            "#\n"
            "# E32 evaluation harness: E14's, with the server on $R/plugrl-server-e32",
        ),
        ("E=$R/e32\n", "E=$R/e36\n"),
        ("RES=${E32_RES:-$E/results}", "RES=${E36_RES:-$E/results}"),
    ],
)

derive(
    "e25_cell.sh",
    "dppo_cell.sh",
    [
        (
            "# E25 on qz103: pi0.5 trained by DPPO on LIBERO-10 task 8, through PlugRL.",
            "# E36's DPPO cell: E25's, with output under e36/, the server on\n"
            "# $R/plugrl-server-e32 (E32's code, which has #58), and the memory recorder\n"
            "# following the cards the server was given. E25's header follows.\n"
            "#\n"
            "# E25 on qz103: pi0.5 trained by DPPO on LIBERO-10 task 8, through PlugRL.",
        ),
        ("E=$R/e25\n", "E=$R/e36\n"),
        ("SRC=$R/plugrl-server-main/src", "SRC=$R/plugrl-server-e32/src"),
        (
            'src=$(cat $R/plugrl-server-main/COMMIT)"',
            'src=$(cat $R/plugrl-server-e32/COMMIT)"',
        ),
        (
            '  cd "$R/plugrl-server-main" && exec setsid env \\',
            '  cd "$R/plugrl-server-e32" && exec setsid env \\',
        ),
        (
            'mkdir -p "$OUT"\n',
            'mkdir -p "$OUT"\n_SG="${SRV_GPUS:-0,1}"\nexport MEM_G0="${_SG%%,*}" MEM_G1="${_SG##*,}"\n',
        ),
        (
            "--format=csv,noheader,nounits -i 0),$(nvidia-smi --query-gpu=memory.used "
            "--format=csv,noheader,nounits -i 1)",
            "--format=csv,noheader,nounits -i $MEM_G0),$(nvidia-smi --query-gpu=memory.used "
            "--format=csv,noheader,nounits -i $MEM_G1)",
        ),
        (
            'log "peak MiB on cards 0 and 1:',
            'log "peak MiB on the server cards $MEM_G0 and $MEM_G1:',
        ),
        ('log "E25_CELL_DONE $CELL"', 'log "E36_DPPO_DONE $CELL"'),
    ],
)
