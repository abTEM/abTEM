(library:contributing)=

# Contribute

Welcome to the *ab*TEM repository! We’re excited you’re here and wish to contribute.

There are many ways you can help the community improve *ab*TEM, with some main ones being:

* Bug reports and enhancement requests
* Add your example to the gallery
* Contribute to the documentation
* Contribute to the code base

## Bug reports and enhancement requests

Bug reports are an important part of keeping *ab*TEM a stable, robust code. Bug reports should be submitted as a GitHub 
issue, sharing them with the *ab*TEM community for comments/ideas from others. 

Before creating an issue, it is good practice to search existing issues and pull requests to see whether the issue has 
already been reported and/or fixed. Since *ab*TEM is in active development, you should also always check whether your
installed version is up to date.

Having a complete report will allow others to reproduce the bug and provide insight into fixing it. In short, a good
bug report should:

1. Include a short, self-contained Python snippet reproducing the
   problem, see [this article](https://stackoverflow.com/help/minimal-reproducible-example) for tips. You can format the
   code nicely by using [GitHub Flavored Markdown](https://docs.github.com/en/get-started/writing-on-github):
   ````
   ```python
   # your code goes here
   ```
   ````
2. Include the version of *ab*TEM and any relevant dependencies.

3. Explain why the current behavior is wrong/not desirable, and what you expect instead.

## The example gallery

Contributing an example to the gallery can be done in a few steps without any additional installations:

1. Add a notebook to the `user_guide/examples/notebooks` folder and a thumbnail to the `user_guide/examples/thumbnails` folder.
2. Update the `user_guide/examples/examples.yml` file with your entry.
3. Regenerate the `user_guide/examples/examples.md` index file using a script (to be added) and open a pull request.

```{tip}
Add the code for generating the thumbnail as the last cell of the example notebook. You can use the tags `hide-input`
and `remove-cell` to hide the cell from the documentation rendition of the example 
([how to add metadata to notebooks](https://jupyterbook.org/en/stable/content/metadata.html)).
```

## Contributing to the documentation

The documentation exists in two places, under `/docs` in the form of markdown files and Jupyter notebooks, and as
docstrings in the code library itself, which become the auto-generated [API reference](reference:api).

The documentation is built using [`jupyter-book`](https://jupyterbook.org/en/stable/intro.html). To generate the docs
from the source, navigate to the `/docs` folder and run `jb build .` at the command line. The built site will be
available in your `_build\html` folder.

The online documentation is automatically updated when the source is updated. 

## Getting started with the codebase

To get started with the *ab*TEM codebase, take the following steps:

(constributing:clone_and_install)
### Clone and install

Clone the repository:

```
git clone https://github.com/abtem/abtem
cd abtem
```

Next, install:

```{code-block}
python -m pip install -e .[testing,docs] 

```

This will install *ab*TEM locally in your Python environment (we recommend using Conda), along with the packages needed
to test it and the packages for producing documentation.

### Optional: Install the pre-commit hooks

*ab*TEM uses pre-commit to ensure code style before a commit is made. This ensures that the look and
feel of the code remains consistent over time and across developers. We use the following tools:

* [Black](https://black.readthedocs.io/en/stable/) for standardized code formatting
* [blackdoc](https://blackdoc.readthedocs.io/en/latest/) for standardized code formatting in documentation
* [Flake8](https://flake8.pycqa.org/en/latest/) for general code quality
* [isort](https://pycqa.github.io/isort/) for standardized order in imports (see also [flake8-isort](https://github.com/gforcada/flake8-isort)).

To enable pre-commit for your clone, run the following from the repository root:

```{code-block}
pre-commit install
```

From now on, when you make a commit to *ab*TEM, pre-commit will ensure that your code looks correct according to a few
checks.

### Run the tests

For code testing, *ab*TEM uses `pytest`. The test suite also relies on [`hypothesis`](https://hypothesis.readthedocs.io/en/latest/) for property-based testing. You
may run all the tests, or only ones that do not have a specific mark, with the following command:

```{code-block}
pytest
```

You may exclude tests that require additional installations and tests that generally runs slow, for example:

```{code-block}
pytest -m 'not requires_gpaw and not requires_gpu and not test_slow'
```

You can alternatively use [tox](https://tox.wiki/en/latest/) to run the tests in multiple isolated environments, and also without the need for the
initial dependencies install (see the `tox.ini` file for available test environments and further explanation):

```{code-block}
tox -e py39-sphinx4 -- -m 'not requires_chrome and not requires_tex'
```