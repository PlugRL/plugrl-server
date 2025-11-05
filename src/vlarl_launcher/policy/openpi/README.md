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