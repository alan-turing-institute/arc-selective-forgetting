import math
from typing import Iterable

import datasets
import numpy as np
from datasets import Dataset, load_dataset
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

from arcsf.utils import hf_progress_bars_disabled

TOFU_PATH = "locuslab/TOFU"
TOFU_SUBSET = "full"
TOFU_NUM_AUTHORS = 200
TOFU_Q_PER_AUTHOR = 20
TOFU_BIO_Q_PER_AUTHOR = 4


def _get_forget_index(author: int, question: int, TOFU_Q_PER_AUTHOR: int) -> int:
    """
    returns the question index in the dataset, given an author number and question
    number (within the author) (inner function)
    """
    return question + TOFU_Q_PER_AUTHOR * author


def get_forget_indices(
    authors: int | Iterable[int], questions: int | Iterable[int], TOFU_Q_PER_AUTHOR: int
) -> list[int]:
    """
    Returns the question indices in the dataset, given author numbers and within-author
    question numbers (across dataset)
    """
    if isinstance(authors, int):
        authors = [authors]
    if isinstance(questions, int):
        questions = [questions]

    return [
        _get_forget_index(author, question, TOFU_Q_PER_AUTHOR)
        for author in authors
        for question in questions
    ]


def load_tofu(
    granularity: str,
    random_seed: int,
    stratified: bool,
    forget_random: bool,
    forgotten_author_fraction: float,
    forgotten_fact_fraction: float,
) -> tuple[Dataset, Dataset] | tuple[None, Dataset]:
    """
    Loads TOFU dataset given different flags for retain--forget split.
    Args:
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
    if granularity not in ["question", "author"]:
        raise ValueError(
            f"granularity is {granularity} but must be one of question or author"
        )

    all_data = load_dataset(TOFU_PATH, TOFU_SUBSET)["train"]

    q_indices = [q for q in range(TOFU_NUM_AUTHORS * TOFU_Q_PER_AUTHOR)]
    a_indices = [math.floor(q / TOFU_Q_PER_AUTHOR) for q in q_indices]

    all_data = all_data.add_column("question_index", q_indices)
    all_data = all_data.add_column("author_index", a_indices)

    # if the fractions are both 0, return the full dataset and None
    if forgotten_fact_fraction == 0.0 and forgotten_author_fraction == 0.0:
        return None, all_data

    # if author, then the number of questions to forget (per author) is all of them
    if granularity == "author":
        num_forget = TOFU_Q_PER_AUTHOR
    # else: it depends on other arguments
    elif granularity == "question":
        if stratified:
            # non-random stratification: hard code 4 questions for now
            if not forget_random:
                num_forget = int(TOFU_BIO_Q_PER_AUTHOR)
            # random then it depends on TOFU_Q_PER_AUTHOR and forgotten_fact_fraction
            elif forget_random:
                num_forget = int(TOFU_Q_PER_AUTHOR * forgotten_fact_fraction)
        # non stratified random: forgotten_fact_fraction applied across entire dataset
        elif not stratified:
            num_forget = int(
                TOFU_NUM_AUTHORS * TOFU_Q_PER_AUTHOR * forgotten_fact_fraction
            )

    # Get full list of indices
    all_indices = list(range(all_data.num_rows))

    # Get full list of authors
    authors = list(range(TOFU_NUM_AUTHORS))

    # Conditionally create full list of questions
    if granularity == "author" or not forget_random:
        forget_questions = list(range(num_forget))

    # create lists of retained and forgotten authors in all cases (sometimes not used)
    retain_authors, forget_authors = train_test_split(
        authors,
        test_size=forgotten_author_fraction,
        random_state=random_seed,
    )

    if granularity == "question":
        if forget_random:
            if stratified:
                # if stratified and random: select train-test split within each
                # set of author questions (currently selects same indices)
                forget_indices = []
                for author in forget_authors:
                    all_author_indices = get_forget_indices(
                        author, range(TOFU_Q_PER_AUTHOR), TOFU_Q_PER_AUTHOR
                    )

                    _, author_forget_indices = train_test_split(
                        all_author_indices,
                        test_size=num_forget,
                        random_state=random_seed,
                    )
                    forget_indices.append(author_forget_indices)
                forget_indices = np.concatenate(forget_indices).tolist()
                retain_indices = list(set(all_indices).difference(forget_indices))

            # if not stratified we can apply train_test_split across entire dataset
            else:
                retain_indices, forget_indices = train_test_split(
                    all_indices,
                    test_size=num_forget,
                    random_state=random_seed,
                )
        # if not random then take first num_forget questions
        # forget_questions contains the relevant indices for this
        else:
            forget_indices = get_forget_indices(
                forget_authors, forget_questions, TOFU_Q_PER_AUTHOR
            )
            retain_indices = list(set(all_indices).difference(forget_indices))

    # if granularity == author we just use retain/forget authors to define our splits
    elif granularity == "author":
        retain_indices = get_forget_indices(
            retain_authors, forget_questions, TOFU_Q_PER_AUTHOR
        )
        forget_indices = get_forget_indices(
            forget_authors, forget_questions, TOFU_Q_PER_AUTHOR
        )

    # create datasets
    forget_set, retain_set = Dataset.from_dict(
        all_data[forget_indices]
    ), Dataset.from_dict(all_data[retain_indices])

    return forget_set, retain_set


class TofuPerturber:
    """
    Function for retrieving perturbed (erroneous) samples at evaluation time
    """

    def __init__(self, data: datasets.Dataset, n_perturbed: int, random_seed: int):
        """
        Intialising the perturbing class

        Args:
            data: dataset from which the samples are being perturbed
            n_perturbed: number of perturbed samples the __call__ function should output
            random_seed: seed for the random element, so experiments are repeatable
        """
        self.data = data
        self.n_perturbed = n_perturbed
        self.rand_gen = default_rng(random_seed)

    def __call__(self, idx: int) -> list[str]:
        """
        Args:
            idx: index from the dataset from which the sample is being perturbed

        Returns:
            list of strings representing perturbed samples
        """
        # Perturbed answer: Incorrect answer to this question (here pick random answers
        # from a different question about the same author)
        author_n = self.data[idx]["author_index"]
        question_n = self.data[idx]["question_index"]
        # this disables progress bars appearing on every __call__
        with hf_progress_bars_disabled():
            perturbed_options = self.data.filter(
                lambda sample: sample["author_index"] == author_n
                and sample["question_index"] != question_n
            ).shuffle(generator=self.rand_gen)
        # To ensure we aren't sampling more perturbed options than there are available
        if len(perturbed_options) < self.n_perturbed:
            raise ValueError(
                f"{self.n_perturbed=} but only {len(perturbed_options)} possible "
                "perturbed answers are available."
            )
        # return the number was want
        perturbed_options = perturbed_options[: self.n_perturbed]["answer"]
        return perturbed_options
