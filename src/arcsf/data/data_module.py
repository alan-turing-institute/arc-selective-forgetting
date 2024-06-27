import copy
from importlib import resources
from typing import Any, Callable, Dict, List

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers.data.data_collator import InputDataClass

import arcsf.data
from arcsf.data.tofu import load_tofu
from arcsf.utils import hf_progress_bars_disabled

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


class QAFormatter:
    """
    Formats question-answer pairs into a single string using a template.
    """

    def __init__(self, template: str):
        """
        Args:
            template: A string template containing "{question}", "{answer}", and
                "{eos_token}"
        """
        for key in ["{question}", "{answer}", "{eos_token}"]:
            if key not in template:
                raise ValueError(
                    "Template must contain '{question}', '{answer}' and '{eos_token}'."
                )
        self.template = template

    def __call__(self, question: str, answer: str, eos_token: str | None) -> str:
        """
        Formats a question-answer pair using the template.

        Args:
            question: Question to format
            answer: Answer to format
            eos_token: End-of-sequence token to append to the formatted string (set to
                an empty string if None)
        """
        if eos_token is None:
            eos_token = ""
        return self.template.format(
            question=question, answer=answer, eos_token=eos_token
        )


class BlankQAFormatter(QAFormatter):
    """
    Simple QAFormatter that separates questions and answers with a space and no
    additional formatting.
    """

    def __init__(self):
        super().__init__("{question} {answer}{eos_token}")


class EvalQADataset(torch.utils.data.Dataset):
    """
    Question answer format dataset, __getitem__ returns a tokenized question--answer
    pair as a tuple. There is an option to output the answers using "I don't know"
    synonyms by specifying loss_type as "idk".
    """

    def __init__(
        self,
        data: datasets.Dataset,
        tokenizer: AutoTokenizer,
        qa_formatter: QAFormatter,
        loss_type: str,
        quantitative_eval: bool = True,
        qualitative_eval: bool = False,
        device: torch.device = torch.device("cpu"),
        random_seed: int = 42,
        **kwargs,
    ):
        """
        Dataset for evaluation purposes which returns a tokenized version of the input
        and target given a tokenizer and question-answer formatting function. Currently
        outputs each separately, however this can be changed at a later date.

        Args:
            data: torch Dataset containing data for the dataset
            tokenizer : Used to tokenize the input
            qa_formatter : QAFormatter instance used to format input before passing it
                to the model
            loss_type : type of loss used, currently only one option changes behaviour:
                    "idk" : labels are sampled from 'idk.jsonl'
            return_perturbed : Flag denoting whether returning perturbed samples for
                    eval effects the format function, adding perturbed samples
            device : torch.device to pass inputs to
            random_seed: random seed for sampling the retain and idk samples, if used
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
        self.rand_gen = torch.Generator().manual_seed(random_seed)
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.data = data
        self.device = device

        if "n_perturbed" in kwargs.keys():
            self.n_perturbed = kwargs["n_perturbed"]
        else:
            self.n_perturbed = 2

        if loss_type == "idk":
            self.idk = get_idk_responses()
            self.answer_sampler = self.get_idk
        else:
            self.answer_sampler = self.get_answer

        if qualitative_eval and quantitative_eval:
            self.max_length = tokenizer.model_max_length
            self.format = self.eval_script_formatter
        elif qualitative_eval:
            self.format = self.qualitative_formatter
        else:
            self.max_length = tokenizer.model_max_length
            self.format = self.get_perturbed

    def get_idk(self, _):
        """returns randomly sampled "I don't know" answer"""
        rand_pos = torch.randint(0, len(self.idk), (1,), generator=self.rand_gen).item()
        return self.idk[rand_pos]

    def get_answer(self, row):
        """returns answer from a given row"""
        return self.data[row]["answer"]

    def qualitative_formatter(self, qa, _):
        encoded_inp = self.tokenizer(
            qa[0], return_tensors="pt", add_special_tokens=True
        )
        encoded_tar = self.tokenizer(
            qa[1], return_tensors="pt", add_special_tokens=True
        )
        return (encoded_inp, encoded_tar)

    def model_formatter(
        self,
        qa: tuple[str, str],
    ) -> dict[str : torch.Tensor]:
        """Formats the question-answer pair in appropriate format for batch computation.

        Args:
            qa : Tuple containing a question--answer pair

        Returns:
            formatted : formatted version of the input which can readily be passed to
            model
        """
        question, answer = qa
        encoded = self.tokenizer(
            self.qa_formatter(question, answer, self.tokenizer.eos_token),
            max_length=self.max_length,
            truncation=False,
        )
        num_question_tokens = len(self.tokenizer.tokenize(question))
        label = copy.copy(encoded["input_ids"])
        # change label to -100 for question tokens
        for i in range(num_question_tokens):
            label[i] = -100
        return {
            "input_ids": torch.tensor(encoded["input_ids"]).to(self.device),
            "labels": torch.tensor(label).to(self.device),
            "attention_mask": torch.tensor(encoded["attention_mask"]).to(self.device),
        }

    def get_perturbed(
        self, qa: tuple[str, str], row: int
    ) -> list[dict[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Returns batch-formatted version of the question answer pair along with
            perturbed, erroneous question-answer pairs. For evaluation purposes.

        Args:
            qa : ground truth question answer tuple
            row : question row ID for obtaining perturbed input

        Returns:
            formatted: batch formatted version of the perturbed inputs along with
                formatted ground truth inputs
        """
        inp, _ = qa
        author_n = self.data[row]["author_index"]
        question_n = self.data[row]["question_index"]

        with hf_progress_bars_disabled():
            perturbed_options = self.data.filter(
                lambda sample: sample["author_index"] == author_n
                and sample["question_index"] != question_n
            ).shuffle(seed=self.random_seed)

        perturbed_options = perturbed_options[: self.n_perturbed]["answer"]

        formatted = [
            self.model_formatter((inp, perturbed)) for perturbed in perturbed_options
        ]
        return [self.model_formatter(qa)] + formatted

    def eval_script_formatter(self, qa, idx):
        (formatted_question, formatted_answer) = self.qualitative_formatter(qa, idx)
        return self.get_perturbed(qa, idx), (
            formatted_question.to(self.device),
            formatted_answer.to(self.device),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp = self.data[idx]["question"]
        tar = self.answer_sampler(idx)

        return self.format((inp, tar), idx)


class FinetuneDataset(torch.utils.data.Dataset):
    """
    Finetune version of the dataset, __getitem__ returns a sample taken either from
    retain, forget subsets, or a combination of both. Samples are formatted using a
    question formatter allowing for autoregression.
    """

    def __init__(
        self,
        data: datasets.Dataset,
        tokenizer: AutoTokenizer,
        qa_formatter: QAFormatter,
    ):
        """
        Dataset which returns a tokenized version of the input given a tokenizer and
        question-answer formatting function.

        Args:
            data: torch Dataset containing data for the dataset
            tokenizer : Used to tokenize the input
            qa_formatter : QAFormatter instance used to format input before passing it
                to the model
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.qa_formatter = qa_formatter
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        question = self.data[idx]["question"]
        answer = self.data[idx]["answer"]

        inp = self.qa_formatter(question, answer, self.tokenizer.eos_token)

        return self.tokenizer(inp)


class QAForgetDataset(torch.utils.data.Dataset):
    """
    Q+A Forget version of the dataset, __getitem__ returns a retain and forget sample.
    Both are formatted using a question formatter. There is an option to output samples
    using "I don't know" synonyms by specifying loss_type as "idk".
    """

    def __init__(
        self,
        data: tuple[datasets.Dataset, datasets.Dataset],
        tokenizer: AutoTokenizer,
        qa_formatter: QAFormatter,
        loss_type: str,
        random_seed: int = 42,
    ):
        """
        Dataset which returns a tokenized version of the input given a tokenizer and
        question-answer formatting function.

        Args:
            data: tuple of forget and retain Datasets
            tokenizer : Used to tokenize the input
            qa_formatter : QAFormatter used to format input before passing it to the
                model
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
        return len(self.forget_data)

    def __getitem__(self, idx):
        # forget data works as in above examples
        forget_row = self.forget_data[idx]
        forget_question = forget_row["question"]
        forget_answer = self.answer_sampler(forget_row)

        # this takes the first item in our retain data permutation using item_index
        retain_row = self.retain_data[self.item_index]
        # then iterates the item_index to ensure a new sample is chosen next time
        self.item_index = (self.item_index + 1) % self.retain_length
        retain_question = retain_row["question"]
        retain_answer = retain_row["answer"]

        forget = self.qa_formatter(
            forget_question, forget_answer, self.tokenizer.eos_token
        )
        retain = self.qa_formatter(
            retain_question, retain_answer, self.tokenizer.eos_token
        )

        return self.tokenizer(forget), self.tokenizer(retain)


class ForgetterDataCollator:
    """
    Data collator that parses lists of forget and retain inputs as provided by
    QAForgetDataset.
    """

    def __init__(self, base_collator: Callable[[List[InputDataClass]], Dict[str, Any]]):
        """
        Args:
            base_collator: An instance of a normal HuggingFace (or custom) data collator
                which takes a list of model inputs and collates them into a batch.
        """
        self.base_collator = base_collator

    def __call__(
        self, features: List[Dict[str, Any]], **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Args:
            features: A list of outputs from QAForgetDataset, containing tuples of
                forget and retain data.
            kwargs: Additional arguments to pass to the base collator.

        Returns:
            Batch of forget and retain inputs.
        """
        forget = self.base_collator([sample[0] for sample in features], **kwargs)
        retain = self.base_collator([sample[1] for sample in features], **kwargs)

        return forget, retain


class EvaluationCollateFunction:
    def __init__(self, padding_value, padding_side="left", batch_first=True):
        if isinstance(padding_value, int):
            self.padding_value = padding_value
        else:
            raise ValueError("padding_value must be an integer")

        if padding_side == "left":
            self.reverse = lambda x: torch.flip(x, [-1])
        else:
            self.reverse = lambda x: x

        self.batch_first = batch_first

    def __call__(self, batch):
        logit_inputs = [inp for (inp, _) in batch]
        qa_pairs = [question_answer_pair for (_, question_answer_pair) in batch]
        n_perturbed = len(logit_inputs[0])
        batch_input = [
            {
                "input_ids": self.reverse(
                    pad_sequence(
                        [self.reverse(inp[idx]["input_ids"]) for inp in logit_inputs],
                        batch_first=self.batch_first,
                        padding_value=self.padding_value,
                    )
                ),
                "labels": self.reverse(
                    pad_sequence(
                        [self.reverse(inp[idx]["labels"]) for inp in logit_inputs],
                        batch_first=self.batch_first,
                        padding_value=-100,
                    )
                ),
                "attention_mask": self.reverse(
                    pad_sequence(
                        [
                            self.reverse(inp[idx]["attention_mask"])
                            for inp in logit_inputs
                        ],
                        batch_first=self.batch_first,
                        padding_value=0,
                    )
                ),
            }
            for idx in range(n_perturbed)
        ]
        questions = {
            "input_ids": self.reverse(
                pad_sequence(
                    [
                        self.reverse(question["input_ids"][0])
                        for (question, _) in qa_pairs
                    ],
                    batch_first=self.batch_first,
                    padding_value=self.padding_value,
                )
            ),
            "attention_mask": self.reverse(
                pad_sequence(
                    [
                        self.reverse(question["attention_mask"][0])
                        for (question, _) in qa_pairs
                    ],
                    batch_first=self.batch_first,
                    padding_value=0,
                )
            ),
        }

        answers = {
            "input_ids": self.reverse(
                pad_sequence(
                    [self.reverse(answer["input_ids"][0]) for (_, answer) in qa_pairs],
                    batch_first=self.batch_first,
                    padding_value=self.padding_value,
                )
            ),
            "attention_mask": self.reverse(
                pad_sequence(
                    [
                        self.reverse(answer["attention_mask"][0])
                        for (_, answer) in qa_pairs
                    ],
                    batch_first=self.batch_first,
                    padding_value=0,
                )
            ),
        }

        return batch_input, (questions, answers)


# in progress refactor #


class EvaluateCollateFunction:
    def __init__(self, tokenizer):
        self.padding_data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding=True
        )

    def __call__(self, batch):
        logit_inputs = [inp for (inp, _) in batch]
        qa_pairs = [question_answer_pair for (_, question_answer_pair) in batch]
        n_perturbed = len(logit_inputs[0])
        batch_input = [
            self.padding_data_collator([inp[idx] for inp in logit_inputs])
            for idx in range(n_perturbed)
        ]
        questions = self.padding_data_collator([question for (question, _) in qa_pairs])
        answers = self.padding_data_collator([answer for (_, answer) in qa_pairs])

        return batch_input, (questions, answers)
