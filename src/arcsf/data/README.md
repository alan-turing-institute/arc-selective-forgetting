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
There are 4 possible granularity options, defined using three input arguments `granularity`, `forget_random`, and `stratified`.

The first output is the forget set, and the second is the retain set.

It can (hopefully) be integrated into the source code in the following way, using the Text:

    class TextForgetDatasetQA(Dataset):
        def __init__(self, data_path, tokenizer, model_family,  max_length=512, split = "forget10", loss_type="idk"):
            super(TextForgetDatasetQA, self).__init__()
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.forget_data = datasets.load_dataset(data_path, split)["train"] # <<<< Replace with the `forget` output of load_tofu
            retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
            self.retain_data =datasets.load_dataset(data_path, retain_split)["train"] # <<<< Replace with the `retain` output of load_tofu
            self.model_configs = get_model_identifiers_from_yaml(model_family)
            self.loss_type = loss_type

            ...

The `self.forget_data` and `self.retain_data` attributes should be replaced with those defined using `load_tofu`. There is no need to specify the `['train']` split, since the `load_tofu` function does not create a datasetDict (though there is scope to change this if necessary for improved integrability).


## Data Module
### `QAForgetSet`

This is a `torch.utils.data.dataset` class which contains question answer pairs for forgetting. It has 4 attributes:
- `tokenizer`: This is the tokenizer which is used convert the raw text into tokens
- `loss_type`: This is the type of loss being used. If the desired outcome is "I don't know" or something to  that effect use the argument - `'idk'`.
- `forget_data`: This is the forget data entries defined using `load_tofu`.
- `idk`: If the loss_type is set to `'idk'` then this attribute is defined to correspond to a list of possible strings which are  used to replace the target answers.

There are also a number of optional arguments in the `__init__()`, most of which are passed into the `load_tofu` call:
  - `granularity`: Corresponds to the level of granularity in the forget set.
  - `a_to_drop`: Defaults to 0.1, corresponds to the fraction of authors to randomly drop.
  - `q_to_drop`: Defaults to 0.1, corresponds to the fraction of questions to randomly drop within author question sets.
  - `random_seed`: Random seed for repeatability, defaults to 42, the answer to life.
  - `tokenizer`: As above, the tokenizer should be passed to the class at initialisation.
  - `loss_type`: As above, the type loss is defined in the class initialisation.
