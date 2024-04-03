# Utils
## Data Utils 
### `load_tofu`
 `load_tofu` is used to load the TOFU datasets compatible with the original source code, but with varying forget/retain sizes, and varying granularity (pending) of unlearning content.

 It outputs two datasets of the format/type:
    
    Dataset({
        features: ['question', 'answer'],
        num_rows: N # N depends on the chosen forget fraction
    })

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

The self.forget_data and self.retain_data attributes should be replaced with those defined using `load_tofu`. There is no need to specify the `['train']` split, since the `load_tofu` function does not create a datasetDict (though there is scope to change this if necessary for improved integrability). 