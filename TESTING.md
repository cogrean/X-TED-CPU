# Running Tests

All tests assume the package is installed in your environment:

```bash
pip install x-ted
```

Or from source:

```bash
git clone https://github.com/cogrean/X-TED-CPU.git
cd X-TED-CPU
pip install .
```

## Smoke test

A broad integration test that exercises the core API, batch computation, and NLP functions against reference implementations (zss). Requires `zss` and the `nlp` extra.

```bash
pip install zss
pip install x-ted[nlp]
python -m spacy download en_core_web_sm
python tests/test_smoke.py
```

## NLP tests

Unit tests for `x_ted_util_transfer` and `x_ted_compute_from_text`, including graceful error handling when spaCy is not installed. Requires `pytest` and the `nlp` extra.

```bash
pip install pytest
pip install x-ted[nlp]
python -m spacy download en_core_web_sm
pytest tests/test_nlp.py
```

## Running all pytest tests

```bash
pytest tests/
```
