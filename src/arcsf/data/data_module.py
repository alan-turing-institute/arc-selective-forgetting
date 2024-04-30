import torch
from torch.utils.data import ConcatDataset, Dataset

from arcsf.data.tofu import load_tofu

_dataset_dict = {"tofu": load_tofu}


def get_data(
    dataset_name,
    granularity,
    stratified,
    forget_random,
    forgotten_author_fraction,
    forgotten_fact_fraction,
    random_seed,
):
    load_func = _dataset_dict[dataset_name]
    data = load_func(
        granularity,
        stratified,
        forget_random,
        forgotten_author_fraction,
        forgotten_fact_fraction,
        random_seed=random_seed,
    )

    return data


def QAformatter_basic(QA: tuple[str]) -> str:
    """
    Basic QA formatter which accepts a tuple outputs:

    "Question: [input question]\nAnswer: [input answer]"

    args:
        - QA: Tuple of question answer pair
    returns:
        - full_text: formatted question--answer pair
    """
    Question, Answer = QA
    question_formatting = "Question: " + Question
    answer_formatting = "\nAnswer: " + Answer
    full_text = question_formatting + answer_formatting
    return full_text


class EvalQADataSet(Dataset):
    """
    Question answer format dataset, __getitem__ returns a tokenized question--answer
    pair as a tuple. There is an option to output the answers using "I don't know"
    synonyms by specifying loss_type as "idk".
    """

    def __init__(
        self,
        data,
        tokenizer,
        qa_formatter,
        split,
        loss_type,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
        self.loss_type = loss_type
        forget_data, retain_data = data

        split_dict = {"retain": retain_data, "forget": forget_data}
        self.data = split_dict[split]

        if loss_type == "idk":
            with open("src/arcsf/data/idk.jsonl") as idk_file:
                self.idk = idk_file.read().splitlines()
            self.answer_sampler = self.idk_sampler
        else:
            self.answer_sampler = self.forget_sampler

    def answer_sampler():
        return None

    def idk_sampler(self, _):
        rand_pos = torch.randint(0, len(self.idk), (1,)).item()
        return self.idk[rand_pos]

    def forget_sampler(self, forget_row):
        return forget_row["answer"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = self.data[idx]["question"]

        target = self.answer_sampler(self.data[idx]["answer"])

        return self.tokenizer(input), self.tokenizer(target)


class FinetuneDataset(Dataset):
    """
    Finetune version of the dataset, __getitem__ returns a sample taken either from
    retain, forget subsets, or a combination of both. Samples are formatted using a
    question formatter allowing for autoregression. There is an option to output
    samples using "I don't know" synonyms by specifying loss_type as "idk".
    """

    def __init__(
        self,
        data,
        tokenizer,
        qa_formatter,
        split,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter

        forget_data, retain_data = data

        split_dict = {
            "retain": retain_data,
            "forget": forget_data,
            "all": ConcatDataset([retain_data, forget_data]),
        }
        self.data = split_dict[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        question = self.data[idx]["question"]
        answer = self.qa_formatter(self.data[idx]["answer"])

        inp = self.qa_formatter((question, answer))

        return self.tokenizer(inp)


class QAForgetDataSet(Dataset):
    """
    Q+A Forget version of the dataset, __getitem__ returns a retain and forget sample.
    Both are formatted using a question formatter. There is an option to output samples
    using "I don't know" synonyms by specifying loss_type as "idk".
    """

    def __init__(
        self,
        data,
        tokenizer,
        qa_formatter,
        loss_type,
        random_seed=42,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
        self.loss_type = loss_type

        self.forget_data, self.retain_data = data

        # shuffle the retain data and get the question indices for debugging
        self.retain_data = self.retain_data.shuffle(seed=random_seed)
        self.retain_permutation = self.retain_data["question_index"]
        # set item index acting as a counter for retain permutation
        self.item_index = 0
        self.retain_length = len(self.retain_data)

        if loss_type == "idk":
            with open("src/arcsf/data/idk.jsonl") as idk_file:
                self.idk = idk_file.read().splitlines()
            self.answer_sampler = self.idk_sampler
        else:
            self.answer_sampler = self.forget_sampler

    def idk_sampler(self, _):
        rand_pos = torch.randint(0, len(self.idk), (1,)).item()
        return self.idk[rand_pos]

    def forget_sampler(self, forget_row):
        return forget_row["answer"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # this takes the first item in our retain data permutation using item_index
        retain_row = self.retain_data[self.item_index]
        # then rolls the permutation vector using the item_index to ensure
        # samples aren't reused without first exhausting all retain samples
        self.item_index = (self.item_index + 1) % self.retain_length

        retain_question = retain_row["question"]
        retain_answer = retain_row["answer"]

        # forget data works as in above examples
        forget_row = self.forget_data[idx]
        forget_question = forget_row["question"]

        forget_answer = self.answer_sampler(forget_row)

        retain = self.qa_formatter((retain_question, retain_answer))
        forget = self.qa_formatter((forget_question, forget_answer))

        return self.tokenizer(retain), self.tokenizer(forget)
