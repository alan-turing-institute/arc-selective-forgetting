import torch
from torch.utils.data import Dataset

from arcsf.data.data_utils import load_tofu


class QADataSet(Dataset):
    def __init__(
        self,
        tokenizer,
        granularity,
        split="forget",
        a_to_drop=0.1,
        q_to_drop=0.1,
        loss_type="standard",
        random_seed=42,
    ):
        super(QADataSet, self).__init__()
        self.tokenizer = tokenizer
        self.loss_type = loss_type

        forget_data, retain_data, self.debug_dict = load_tofu(
            granularity,
            forgotten_author_fraction=a_to_drop,
            forgotten_fact_fraction=q_to_drop,
            random_seed=random_seed,
            debug=True,
        )
        split_dict = {"retain": retain_data, "forget": forget_data}
        self.data = split_dict[split]

        if loss_type == "idk":
            with open("src/arcsf/data/idk.jsonl") as idk_file:
                self.idk = idk_file.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx]["question"]

        if self.loss_type == "idk":
            rand_pos = torch.randint(0, len(self.idk), (1,)).item()
            target = self.idk[rand_pos]
        else:
            target = self.data[idx]["answer"]

        return input, target
