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

            if self.loss_type == "idk":
                self.split1, self.split2 = "idk", "retain"
                self.idontknowfile = "data/idontknow.jsonl"
                self.idk = open(self.idontknowfile, "r").readlines()
            else:
                self.split1, self.split2 = "forget", "retain"

        def __len__(self):
            return len(self.forget_data)

        def __getitem__(self, idx):
            rets = []
            for data_type in [self.split1, self.split2]:
                #use questions from forget set if split is idk or forget
                data = self.retain_data if data_type == "retain" else self.forget_data
                idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
                question = data[idx]['question']
                answer = data[idx]['answer']

                if data_type == "idk":
                    #get a random answer position from idk
                    rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                    answer = self.idk[rand_pos].strip()
                    
                converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
                rets.append(converted_data)
            return rets