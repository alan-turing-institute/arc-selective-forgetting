import torch
from torch.utils.data import Dataset

from arcsf.data.data_utils import load_tofu


class QAForgetSet(Dataset):
    def __init__(
        self,
        tokenizer,
        granularity,
        a_to_drop=0.1,
        q_to_drop=0.1,
        loss_type="standard",
        random_seed=42,
    ):
        super(QAForgetSet, self).__init__()
        self.tokenizer = tokenizer
        self.loss_type = loss_type

        self.forget_data, _, self.debug_dict = load_tofu(
            granularity,
            forgotten_author_fraction=a_to_drop,
            forgotten_fact_fraction=q_to_drop,
            random_seed=random_seed,
            debug=True,
        )

        if loss_type == "idk":
            with open("src/arcsf/data/idk.jsonl") as idk_file:
                self.idk = idk_file.read().splitlines()

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        forget_input = self.forget_data[idx]["question"]
        if self.loss_type == "idk":
            rand_pos = torch.randint(0, len(self.idk), (1,)).item()
            forget_target = self.idk[rand_pos]
        else:
            forget_target = self.forget_data[idx]["answer"]

        return forget_input, forget_target
