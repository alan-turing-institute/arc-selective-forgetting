from collections.abc import Callable
from importlib import resources

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import arcsf.data
from arcsf.data.tofu import load_tofu

_dataset_dict = {"tofu": load_tofu}


def get_idk_responses() -> list[str]:
    """Returns a list of "I don't know"-style responses."""
    idk_file = resources.files(arcsf.data) / "idk.jsonl"
    with idk_file.open() as f:
        return f.read().splitlines()


def get_data(
    dataset_name,
    granularity,
    stratified,
    forget_random,
    forgotten_author_fraction,
    forgotten_fact_fraction,
    random_seed,
):
    """
    Loads dataset from dataset_dict given the specified dataset, formatted according to
    flags for retain--forget split. Some of these are defined specific to the TOFU
    dataset re author/question etc. This can be changed if additional datasets are used.
    Args:
        dataset_name: name of the dataset which should be loaded
        granularity: level at which forgetting takes place (author vs question)
        stratified: if forgetting questions restrain to specific authors?
        forget_random: is forgetting happening randomly within constraints?
        forgotten_author_fraction: fraction of authors from which to forget questions
        forgotten_fact_fraction: fraction of questions to randomly forget.
            if stratified == True represents fraction of Qs in author, if
            stratified == False represents fraction of total Qs
        random_seed: seed for reproducibility
    Returns:
        Two datasets with forget and retain sets
    """
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


def qa_formatter_basic(qa: tuple[str, str]) -> str:
    """
    Basic QA formatter which accepts a tuple outputs:

    "Question: [input question]\nAnswer: [input answer]"

    Args:
        QA: Tuple of question answer pair

    Returns:
        full_text: formatted question--answer pair
    """
    question, answer = qa
    return f"Question: {question}\nAnswer: {answer}"


class EvalQADataset(Dataset):
    """
    Question answer format dataset, __getitem__ returns a tokenized question--answer
    pair as a tuple. There is an option to output the answers using "I don't know"
    synonyms by specifying loss_type as "idk".
    """

    def __init__(
        self,
        data: Dataset,
        tokenizer: AutoTokenizer,
        qa_formatter: Callable[[tuple[str, str]], str],
        loss_type: str,
        random_seed=42,
    ):
        """
        Dataset for evaluation purposes which returns a tokenized version of the input
        and target given a tokenizer and question-answer formatting function. Currently
        outputs each separately, however this can be changed at a later date.

        Args:
            data: torch Dataset containing data for the dataset
            tokenizer : Used to tokenize the input
            qa_formatter : Function used to format input before passing it to the model
            loss_type : type of loss used, currently only one option changes behaviour:
                    "idk" : labels are sampled from 'idk.jsonl'
            random_seed: random seed for sampling the retain and idk samples, if used
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
        self.rand_gen = torch.Generator().manual_seed(random_seed)
        self.loss_type = loss_type
        self.data = data

        if loss_type == "idk":
            self.idk = get_idk_responses()
            self.answer_sampler = self.get_idk
        else:
            self.answer_sampler = self.get_answer

    def get_idk(self, _):
        """returns randomly sampled "I don't know" answer"""
        rand_pos = torch.randint(0, len(self.idk), (1,), generator=self.rand_gen).item()
        return self.idk[rand_pos]

    def get_answer(self, forget_row):
        """returns returns answer from a given row"""
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
    question formatter allowing for autoregression.
    """

    def __init__(
        self,
        data: Dataset,
        tokenizer: AutoTokenizer,
        qa_formatter: Callable[[tuple[str, str]], str],
    ):
        """
        Dataset which returns a tokenized version of the input given a tokenizer and
        question-answer formatting function.

        Args:
            data: torch Dataset containing data for the dataset
            tokenizer : Used to tokenize the input
            qa_formatter : Function used to format input before passing it to the model
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        question = self.data[idx]["question"]
        answer = self.qa_formatter(self.data[idx]["answer"])

        inp = self.qa_formatter((question, answer))

        return self.tokenizer(inp)


class QAForgetDataset(Dataset):
    """
    Q+A Forget version of the dataset, __getitem__ returns a retain and forget sample.
    Both are formatted using a question formatter. There is an option to output samples
    using "I don't know" synonyms by specifying loss_type as "idk".
    """

    def __init__(
        self,
        data: Dataset,
        tokenizer: AutoTokenizer,
        qa_formatter: Callable[[tuple[str, str]], str],
        loss_type: str,
        random_seed: int = 42,
    ):
        """
        Dataset which returns a tokenized version of the input given a tokenizer and
        question-answer formatting function.

        Args:
            data: torch Dataset containing data for the dataset
            tokenizer : Used to tokenize the input
            qa_formatter : Function used to format input before passing it to the model
            loss_type : type of loss used, currently only one option changes behaviour:
                    "idk" : labels are sampled from 'idk.jsonl'
            random_seed: random seed for sampling the retain and idk samples, if used
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
        self.loss_type = loss_type
        self.rand_gen = torch.Generator().manual_seed(random_seed)
        self.forget_data, self.retain_data = data

        # shuffle the retain data and get the question indices for debugging
        self.retain_data = self.retain_data.shuffle(seed=random_seed)
        self.retain_permutation = self.retain_data["question_index"]
        # set item index acting as a counter for retain permutation
        self.item_index = 0
        self.retain_length = len(self.retain_data)

        if loss_type == "idk":
            self.idk = get_idk_responses()
            self.answer_sampler = self.get_idk
        else:
            self.answer_sampler = self.get_answer

    def get_idk(self, _):
        """returns randomly sampled "I don't know" answer"""
        rand_pos = torch.randint(0, len(self.idk), (1,), generator=self.rand_gen).item()
        return self.idk[rand_pos]

    def get_answer(self, forget_row):
        """returns returns answer from a given row"""
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
