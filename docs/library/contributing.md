# Contributing

## Getting started

### Clone and install

```
git clone https://github.com/abtem/abtem
cd abtem
```

Next install:

```{code-block}
python -m pip install -e .[testing,docs,code_style] 
```

This will install abTEM locally along with the packages needed to test it, the packages for producing documentation and
packages for ensuring the code style.

### Install the pre-commit hooks

abTEM uses pre-commit to ensure code style and quality before a commit is made. This ensures that the look and
feel of abTEM remains consistent over time and across developers.

To enable pre-commit for your clone, run the following from the repository root:

```{code-block}
pre-commit install
```

From now on, when you make a commit to abTEM, pre-commit will ensure that your code looks correct according to a few
checks.

### Run the tests

abTEM uses `pytest` as its test runner. In addition to unit testing, abTEM employs property-based testing using
the `hypothesis` library and regression tests using `pytest-regression` and `pytest-notebook`.

The test suite 

