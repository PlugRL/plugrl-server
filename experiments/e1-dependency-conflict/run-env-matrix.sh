#!/usr/bin/env bash
# E1, third pass: can the environment families coexist with EACH OTHER?
#
# The first two passes asked whether one environment can share a Python
# environment with the training stack. Mostly it can, so that framing is dead.
#
# But that was never the situation a paper is actually in. An evaluation matrix
# spans several environment families at once, and a monolithic framework has to
# stand up all of them. So the question that matters is whether the families
# can coexist with each other - and, underneath that, whether the union of
# everything a full matrix needs is installable at all.
#
# If it is not, the argument survives in a stronger form: the boundary is not
# what lets you run one environment, it is what lets you run the matrix.
#
# Resolution only. Cheap, and a metadata-level failure is the strongest kind.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$HERE/.work-matrix"
RESULTS="$HERE/results-matrix"
PY_VERSION="3.11"

PROTOCOL_SHA="c7961da3ca050717ad3a01721f00afa1ed29bf47"
LIBERO_SHA="f3bf9428c0d7113b0b72d1462d9552fca35fcf92"

ENV_BASE='
    "loguru>=0.7.3,<0.8.0",
    "gymnasium>=1.2.0,<2.0.0",
    "dm-tree>=0.1.9,<0.2.0",
    "websockets>=15.0.1,<16.0.0",
    "msgpack>=1.1.1,<2.0.0",
    "tyro>=0.9.31,<0.10.0",
    "plugrl-protocol @ git+https://github.com/PlugRL/plugrl-protocol.git@'"$PROTOCOL_SHA"'",
    "pandas>=2.3.3,<3.0.0",
    "imageio[ffmpeg]>=2.37.2,<3.0.0",
'

X_ATARI='
    "ale-py==0.9",
    "gymnasium[atari]>=1.2.1,<2.0.0",
'
X_CLASSIC='
    "gymnasium[classic-control]>=1.2.1,<2.0.0",
'
X_D4RL='
    "cython<3",
    "d4rl>=1.1,<2.0.0",
    "patchelf>=0.17.2.4,<0.18.0",
    "gymnasium-robotics[mujoco-py]>=1.4.1,<2.0.0",
'
X_ROBOMIMIC='
    "cython<3",
    "d4rl>=1.1,<2.0.0",
    "patchelf>=0.17.2.4,<0.18.0",
    "robomimic==0.3.0",
    "robosuite<1.5.0",
    "PyOpenGL==3.1.4",
'
X_LIBERO='
    "libero @ git+https://github.com/CTP314/LIBERO.git@'"$LIBERO_SHA"'",
'

run_case() {
    local name="$1" deps="$2"
    mkdir -p "$WORK/$name"
    cat > "$WORK/$name/pyproject.toml" <<EOF
[project]
name = "e1m-$name"
version = "0.0.0"
requires-python = ">=$PY_VERSION,<3.12"
dependencies = [
$deps
]

[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"
EOF

    local log="$RESULTS/$name.log" outcome
    { echo "# case: $name"; echo "# date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo
      cat "$WORK/$name/pyproject.toml"; echo; echo "--- uv lock ---"; } > "$log"

    if (cd "$WORK/$name" && uv lock --no-progress) >> "$log" 2>&1; then
        outcome="RESOLVED"
    else
        outcome="FAILED"
    fi
    printf '%-34s %s\n' "$name" "$outcome"
    printf '%s\t%s\n' "$name" "$outcome" >> "$RESULTS/summary.tsv"
}

mkdir -p "$WORK" "$RESULTS"
: > "$RESULTS/summary.tsv"

printf '%-34s %s\n' CASE OUTCOME
printf -- '------------------------------------------\n'

# Singles, to establish that each family is individually fine.
run_case single-atari      "$ENV_BASE$X_ATARI"
run_case single-classic    "$ENV_BASE$X_CLASSIC"
run_case single-d4rl       "$ENV_BASE$X_D4RL"
run_case single-robomimic  "$ENV_BASE$X_ROBOMIMIC"
run_case single-libero     "$ENV_BASE$X_LIBERO"

printf -- '------------------------------------------\n'

# Pairs across families.
run_case pair-atari-d4rl        "$ENV_BASE$X_ATARI$X_D4RL"
run_case pair-atari-libero      "$ENV_BASE$X_ATARI$X_LIBERO"
run_case pair-d4rl-libero       "$ENV_BASE$X_D4RL$X_LIBERO"
run_case pair-robomimic-libero  "$ENV_BASE$X_ROBOMIMIC$X_LIBERO"
run_case pair-robomimic-atari   "$ENV_BASE$X_ROBOMIMIC$X_ATARI"

printf -- '------------------------------------------\n'

# The whole matrix at once: what one monolithic environment would have to hold
# to run every benchmark in the paper.
run_case all-five-families \
    "$ENV_BASE$X_ATARI$X_CLASSIC$X_D4RL$X_ROBOMIMIC$X_LIBERO"

printf -- '------------------------------------------\n'
echo "logs: $RESULTS/"
