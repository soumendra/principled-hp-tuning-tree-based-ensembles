# Set-up and run

(run all commands from inside this directory)

If you don't already have poetry, you can install it with `pip install poetry`. Then, you can set up the environment with:

```python
poetry install
```

To train all the models:

```python
poetry run python src/rf_tuned.py
```

or

```python
poetry run python src/rf_default.py
```

To start a jupyter notebook for `hiplot.ipynb`,`

```python
poetry run jupyter notebook
```
