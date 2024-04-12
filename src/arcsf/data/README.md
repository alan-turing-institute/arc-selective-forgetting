# Utils
## Data Utils
### `load_tofu`
`load_tofu` is used to load the TOFU datasets compatible with the original source code, but with varying forget/retain sizes, and varying granularity of unlearning content.

It outputs two datasets of the format/type:
```
Dataset({
    features: ['question', 'answer'],
    num_rows: N # N depends on the chosen forget fraction
})
```

#### Granularity
There are 4 possible granularity options:
- `'author_level'`
  - Adds all questions from randomly chosen authors to the forget set. `forgotten_author_fraction (float)` should be passed.
- `'structured_within_authors'`
  - Adds $n$ questions from randomly chosen authors to the forget set. `forgotten_author_fraction` should be passed, $n$ is currently hard coded to 4.
- `'random_within_authors'`
  - Randomly adds a fraction of questions from randomly chosen authors to the forget set. `forgotten_author_fraction` should be passed as well as `forgotten_fact_fraction`.
- `'random'`
  - Randomly adds a percentage of questions across the entire dataset to the forget set. `forgotten_fact_fraction` should be passed.
