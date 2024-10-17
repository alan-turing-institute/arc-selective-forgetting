# Selective Forgetting

This project builds on the [TOFU paper and codebase (Maini, Feng, Schwarzschild et al., 2024)](https://github.com/locuslab/tofu), which published a dataset for benchmarking approximate unlearning methods - techniques for removing knowledge of a concept from a large language model.

We explore two research questions:

- Does the granularity of the concepts being forgotten impact the quality of forgetting that can be achieved? By granularity here we mean the position in a hierarchy of concepts, for example a book is published by an author, an author writes multiple books, and a publisher publishes multiple authors.

- Does removing a relationship between two entities reduce the performance of the model on unrelated questions about those entities?

To address these we create a new TOFU-inspired question-answer dataset with entities of different types (publishers, authors, books) and with each question-answer pair in the dataset labelled with the entities it refers to.

## Installation/Development

You can `pip` install the dependencies and `arcsf` library with `pip install .` in your preferred virtual environment.

We developed the code with [Poetry](https://python-poetry.org/):

1. Install dependencies with Poetry

   ```bash
   poetry install
   ```

2. Install pre-commit hooks:

   ```bash
   poetry run pre-commit install --install-hooks
   ```

## Usage

The main config files and scripts for running experiments are documented in the [scripts readme](scripts/README.md).
