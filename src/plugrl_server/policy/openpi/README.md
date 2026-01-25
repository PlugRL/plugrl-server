### Installation

```
git submodule clone
cd third_party/openpi
cd packages/openpi_client
pip install -e .
cd ../..
pip install e .
cd ../..
poetry install
```

Replace transformers by

```
cp -r third_party/openpi/src/openpi/models_pytorch/transformers_replace/* "$(python -c 'import site; print(site.getsitepackages()[0])')/transformers/"
```

### Reference Score

#### $\pi_{0.5}$-tiny

- Libero Spatial

| Task Suite ID | Baseline SR | +PPO |
| :---: | :---: | :---: |
| 0 | 0.94 | |
| 1 | 0.92 | |
| 2 | 0.98 | |
| 3 | 0.98 | |
| 4 | 0.88 | |
| 5 | 0.82 | |
| 6 | 0.96 | |
| 7 | 0.80 | |
| 8 | 0.88 | |
| 9 | 0.84 | |
| **Total** | **0.90** | **0.952** |

- Libero Goal

| Task Suite ID | Baseline SR | +PPO |
| :---: | :---: | :---: |
| 0 | 0.90 | |
| 1 | 0.90 | |
| 2 | 0.78 | |
| 3 | 0.46 | |
| 4 | 0.94 | |
| 5 | 0.98 | |
| 6 | 0.84 | |
| 7 | 1.0 | |
| 8 | 0.94 | |
| 9 | 0.58 | |
| **Total** | **0.832** | **0.93** |

- Libero Object

| Task Suite ID | Baseline SR | +PPO |
| :---: | :---: | :---: |
| 0 | 0.96 | |
| 1 | 0.90 | |
| 2 | 1.00 | |
| 3 | 0.94 | |
| 4 | 0.88 | |
| 5 | 0.94 | |
| 6 | 0.88 | |
| 7 | 0.92 | |
| 8 | 0.94 | |
| 9 | 0.94 | |
| **Total** | **0.93** | **0.99** |

- Libero 10

| Task Suite ID | Baseline SR | +PPO |
| :---: | :---: | :---: |
| 0 | 0.52 | |
| 1 | 0.46 | |
| 2 | 0.64 | |
| 3 | 0.76 | |
| 4 | 0.36 | |
| 5 | 0.62 | |
| 6 | 0.52 | |
| 7 | 0.82 | |
| 8 | 0.06 | |
| 9 | 0.34 | |
| **Total** | **0.51** | **0.752** |
