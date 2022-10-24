(library:contributing)=

# Contributing

Welcome to the `abTEM` repository! We’re excited you’re here and want to contribute.

There are many ways you can help the community to improve `abTEM`, some of the main ways are:

* Bug reports and enhancement requests
* Add your example to the gallery
* Contributing to the documentation
* Contributing to the code base

## Bug reports and enhancement requests

Bug reports are an important part of making `abTEM` more stable. Bug reports should be submitted as a GitHub issue, the
issue will then show up to the `abTEM` community and be open to comments/ideas from others. Before creating an issue, it
is good practice to search existing issues and pull requests to see if the issue has already been reported and/or fixed.

Having a complete bug report will allow others to reproduce the bug and provide insight into fixing. In short, a good
bug report should:

1. Include a short, self-contained Python snippet reproducing the
   problem, [this article](https://stackoverflow.com/help/minimal-reproducible-example) for tips. You can format the
   code nicely by using [GitHub Flavored Markdown](https://docs.github.com/en/get-started/writing-on-github):
   ````
   ```python
   # your code goes here
   ```
   ````
2. Include the version of `abTEM` and any relevant dependencies.

3. Explain why the current behavior is wrong/not desired and what you expect instead.

## The example gallery

Contributing an example to the gallery can be done in a few steps without any additional installations:

1. Add a notebook to examples folder and a thumbnail to the example thumbnails folder.
2. Update this gallery.yml file with your entry.
3. Regenerate the examples.md index file using this script and open a pull request.

```{tip}
Add the code for generating the thumbnail as the last cell of the example notebook. You can use the tags `hide-input`
and `remove-cell` to hide the cell from the documentation rendition of the example 
([how to add metadata to notebooks](https://jupyterbook.org/en/stable/content/metadata.html)).
```

## Contributing to the documentation

The documentation exists in two places, under `/docs` in the form of markdown files and Jupyter notebooks and as
docstrings in the library which become the auto-generated API reference.

The documentation is build using [`jupyter-book`](https://jupyterbook.org/en/stable/intro.html). To generate the docs
from the source navigating to the `/docs` folder and running `jb build .` at the command line. The built site will be
available in your `_build\html` folder.

The online documentation is automatically updated when the source is updated. 

## Getting started with the codebase

To get started with `abTEM`'s codebase, take the following steps:

### Clone and install

Clone the repository:

```
git clone https://github.com/abtem/abtem
cd abtem
```

Next, install:

```{code-block}
python -m pip install -e .[testing,docs,dev] 
```

This will install abTEM locally along with the packages needed to test it, the packages for producing documentation and
some development packages.

### Optional: Install the pre-commit hooks

abTEM uses pre-commit to ensure code style before a commit is made. This ensures that the look and
feel of *abTEM* remains consistent over time and across developers. We use the following tools:

* Black for standardized code formatting
* blackdoc for standardized code formatting in documentation
* Flake8 for general code quality
* isort for standardized order in imports. See also flake8-isort.

To enable pre-commit for your clone, run the following from the repository root:

```{code-block}
pre-commit install
```

From now on, when you make a commit to `abTEM`, pre-commit will ensure that your code looks correct according to a few
checks.

### Run the tests

For code tests, `abTEM` uses `pytest`, the test suite also relies on [`hypothesis`](https://hypothesis.readthedocs.io/en/latest/) for property based testing. You
may run all the tests: , or only ones that do not have a specific mark, with the following command:

```{code-block}
pytest
```

You may exclude tests that require additional installations and tests that generally runs slow, for example:

```{code-block}
pytest -m 'not requires_gpaw and not requires_gpu and not test_slow'
```

You can alternatively use tox to run the tests in multiple isolated environments, and also without the need for the
initial dependencies install (see the tox.ini file for available test environments and further explanation):

```{code-block}
tox -e py39-sphinx4 -- -m 'not requires_chrome and not requires_tex'
```
