#!/usr/bin/env bash
# The price of rendering LIBERO without a GPU.
#
# Two runs of the same task with the same actions on the same machine. The
# first uses NVIDIA's EGL, as every run in E11 did. The second points glvnd at
# Mesa's EGL and hides every GPU from the process, so rendering falls back to
# the software rasteriser.
#
# Only the environment differs between the two; the Python, the package set
# and the task are identical.
set -uo pipefail

R=/home/gotham/tmp/plugrl
PY=$R/venv-libero-cpu/bin/python
BENCH=$R/render_bench.py

common_env=(
  LIBERO_CONFIG_PATH="$R/.libero"
  XDG_CACHE_HOME="$R/.cache"
  TMPDIR="$R/.tmp"
  MUJOCO_GL=egl
  PYOPENGL_PLATFORM=egl
  BENCH_STEPS="${BENCH_STEPS:-60}"
)

echo "=== hardware EGL, NVIDIA, GPU 1 visible ==="
env "${common_env[@]}" \
    BENCH_LABEL=nvidia-egl \
    CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
    timeout 1800 "$PY" "$BENCH" 2>&1 | grep -E '^BENCH|Error|Traceback' | head -4

# Same variables as above but for the EGL vendor, so the vendor is the only
# difference. LIBERO parses CUDA_VISIBLE_DEVICES as an integer and raises on
# an empty one, so the GPU stays visible; Mesa's software rasteriser does not
# use it. That rendering works with no GPU visible at all is shown separately,
# by a bare mujoco render with the variable unset.
echo "=== software EGL, Mesa swrast ==="
env "${common_env[@]}" \
    BENCH_LABEL=mesa-software \
    CUDA_VISIBLE_DEVICES=1 \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
    timeout 1800 "$PY" "$BENCH" 2>&1 | grep -E '^BENCH|Error|Traceback' | head -4

echo "RENDER_BENCH_DONE"
