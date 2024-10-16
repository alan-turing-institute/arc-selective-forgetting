# Selective Forgetting

This project builds on the [TOFU paper and codebase (Maini, Feng, Schwarzschild et al., 2024)](https://github.com/locuslab/tofu), which published a dataset for benchmarking approximate unlearning methods - techniques for removing knowledge of a concept from a large language model.

We explore two research questions:

- Does the granularity of the concepts being forgotten impact the quality of forgetting that can be achieved? By granularity here we mean the position in a hierarchy of concepts, for example a book is published by an author, an author writes multiple books, and a publisher publishes multiple authors.

- Does removing a relationship between two entities reduce the performance of the model on unrelated questions about those entities?

To address these we create a new TOFU-inspired question-answer dataset with entities of different types (publishers, authors, books) and with each question-answer pair in the dataset labelled with the entities it refers to.

## Usage

**TODO**

## Development

### Developer Setup

1. Install dependencies with Poetry

   ```bash
   poetry install
   ```

2. Install pre-commit hooks:

   ```bash
   poetry run pre-commit install --install-hooks
   ```
