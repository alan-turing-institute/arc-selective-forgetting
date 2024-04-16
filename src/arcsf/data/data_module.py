import torch
from torch.utils.data import ConcatDataset, Dataset

from arcsf.data.data_utils import load_tofu


class QADataSet(Dataset):
    def __init__(
        self,
        tokenizer,
        qa_formatter,
        granularity,
        split="forget",
        a_to_drop=0.1,
        q_to_drop=0.1,
        loss_type="standard",
        random_seed=42,
    ):
        super(QADataSet, self).__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
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

        return self.tokenizer(input), self.tokenizer(target)


class FinetuneDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        qa_formatter,
        granularity,
        split="forget",
        a_to_drop=0.1,
        q_to_drop=0.1,
        loss_type="standard",
        random_seed=42,
    ):
        super(FinetuneDataset, self).__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
        self.loss_type = loss_type

        forget_data, retain_data, self.debug_dict = load_tofu(
            granularity,
            forgotten_author_fraction=a_to_drop,
            forgotten_fact_fraction=q_to_drop,
            random_seed=random_seed,
            debug=True,
        )
        split_dict = {
            "retain": retain_data,
            "forget": forget_data,
            "all": ConcatDataset([retain_data, forget_data]),
        }
        self.data = split_dict[split]

        if loss_type == "idk":
            with open("src/arcsf/data/idk.jsonl") as idk_file:
                self.idk = idk_file.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        question = self.data[idx]["question"]
        if self.loss_type == "idk":
            rand_pos = torch.randint(0, len(self.idk), (1,)).item()
            answer = self.qa_formatter(self.idk[rand_pos])
        else:
            answer = self.qa_formatter(self.data[idx]["answer"])

        inp = self.qa_formatter((question, answer))

        return self.tokenizer(inp)


# should return all
class QAForgetDataSet(Dataset):
    def __init__(
        self,
        tokenizer,
        qa_formatter,
        granularity,
        a_to_drop=0.1,
        q_to_drop=0.1,
        loss_type="standard",
        random_seed=42,
    ):
        super(QAForgetDataSet, self).__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
        self.loss_type = loss_type

        self.forget_data, self.retain_data, self.debug_dict = load_tofu(
            granularity,
            forgotten_author_fraction=a_to_drop,
            forgotten_fact_fraction=q_to_drop,
            random_seed=random_seed,
            debug=True,
        )
        # shuffle the retain data and get the question indices for debugging
        self.retain_data = self.retain_data.shuffle(seed=random_seed)
        self.retain_permutation = self.retain_data["question_index"]
        # set vector containing the current item index in dataloader
        self.item_index = [index for index in range(len(self.retain_data))]

        if loss_type == "idk":
            with open("src/arcsf/data/idk.jsonl") as idk_file:
                self.idk = idk_file.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # this takes the first item in our retain data permutation using item_index
        retain_row = self.retain_data[self.item_index[0]]
        # then rolls the permutation vector using the item_index to ensure
        # samples aren't reused without first exhausting all retain samples
        self.item_index.append(self.item_index.pop(0))

        retain_question = retain_row["question"]
        retain_answer = retain_row["answer"]

        # forget data works as in above examples
        forget_row = self.forget_data[idx]
        forget_question = forget_row["question"]

        if self.loss_type == "idk":
            rand_pos = torch.randint(0, len(self.idk), (1,)).item()
            forget_answer = self.idk[rand_pos]
        else:
            forget_answer = forget_row["answer"]

        retain = self.qa_formatter((retain_question, retain_answer))
        forget = self.qa_formatter((forget_question, forget_answer))

        return (
            self.tokenizer(retain),
            retain_row["question_index"],
        ), (
            self.tokenizer(forget),
            forget_row["question_index"],
        )
