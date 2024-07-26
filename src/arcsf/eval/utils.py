import warnings

import numpy as np
import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizer


def check_nans(array: np.ndarray | torch.Tensor, name: str = "") -> None:
    if name != "":
        name = " " + name
    message = f"NaNs found in{name} array of shape: {array.shape}"
    if isinstance(array, np.ndarray):
        if np.isnan(array).any():
            warnings.warn(message)
    elif isinstance(array, torch.Tensor):
        if torch.isnan(array).any():
            warnings.warn(message)


def first_idx(
    array: torch.Tensor, value: int | float, equal: bool, dim: int
) -> torch.Tensor:
    """
    If equal is True returns the indices of the first occurrence of a value
    in a tensor along a given dimension, or 0 if the value is not found.
    If equal is False returns the indices of the first occurrence of *not* the value,
    or 0 if the whole row of the tensor is equal to the value.

    Args:
        array : tensor to search
        value : value to search for
        equal: whether to search for the value, or anything not equal to the value
        dim : dimension to search along

    Returns:
        tensor of indices of the first occurrence of the value
    """
    if equal:
        bools = array == value
    else:
        bools = array != value
    return torch.argmax(bools.to(int), dim=dim)


def extract_qa_for_generate(
    inputs: dict[str, torch.Tensor], tokenizer: PreTrainedTokenizer
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Uses masked labels (generated in EvalQADataset) to extract the questions and answers
    from a batch of formatted & tokenized combined question and answers.

    Args:
        inputs : batch of inputs for the model (tokenized and formatted combined QA)
        tokenizer : tokenizer used to tokenize the inputs.

    Returns:
        batch of left padded question input_ids and attention_mask, and batch of left
        padded answer question input_ids and attention_mask
    """
    if tokenizer.padding_side != "left":
        raise ValueError(
            "tokenizer.padding_side must be left for this function to work as intended"
        )
    collator = DataCollatorWithPadding(tokenizer, padding=True)

    # index of first non-masked token in labels of each sequence should indicate the
    # start of the answer (question prompt is everything before that)
    answer_start_ids = first_idx(inputs["labels"], value=-100, equal=False, dim=-1)
    questions = []
    answers = []
    for idx, start in enumerate(answer_start_ids):
        questions.append(
            {
                "input_ids": inputs["input_ids"][idx, :start],
                "attention_mask": inputs["attention_mask"][idx, :start],
            }
        )
        answers.append(
            {
                "input_ids": inputs["input_ids"][idx, start:],
                "attention_mask": inputs["attention_mask"][idx, start:],
            }
        )

    questions = collator(questions)
    answers = collator(answers)

    # The original combined QA is left padded to make all the QA the same length.
    # After extracting the questions by themselves they then have more padding than
    # necessary. Remove it here.
    n_trim_pad = torch.min(
        first_idx(questions["attention_mask"], value=1, equal=True, dim=-1)
    )
    questions["input_ids"] = questions["input_ids"][:, n_trim_pad:]
    questions["attention_mask"] = questions["attention_mask"][:, n_trim_pad:]

    return questions, answers
