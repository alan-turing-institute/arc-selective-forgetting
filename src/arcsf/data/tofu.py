import math
from typing import Iterable

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


def _get_forget_index(author: int, question: int, q_per_author: int) -> int:
    """
    returns the question index in the dataset, given an author number and question
    number (within the author) (inner function)
    """
    return question + q_per_author * author


def get_forget_indices(
    authors: int | Iterable[int], questions: int | Iterable[int], q_per_author: int
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
        _get_forget_index(author, question, q_per_author)
        for author in authors
        for question in questions
    ]


def load_tofu(
    granularity: str,
    stratified: bool,
    forget_random: bool,
    forgotten_author_fraction: float,
    forgotten_fact_fraction: float,
    random_seed: int,
) -> tuple[Dataset, Dataset, dict] | tuple[Dataset, Dataset]:
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

    all_data = load_dataset("locuslab/TOFU", "full")["train"]

    num_authors = 200  # hard coding author count for now
    q_per_author = 20  # hard coding author question count for now

    q_indices = [q for q in range(num_authors * q_per_author)]
    a_indices = [math.floor(q / q_per_author) for q in q_indices]

    all_data = all_data.add_column("question_index", q_indices)
    all_data = all_data.add_column("author_index", a_indices)

    # if author, then the number of questions to forget (per author) is all of them
    if granularity == "author":
        num_forget = q_per_author
    # else: it depends on other arguments
    elif granularity == "question":
        if stratified:
            # non-random stratification: hard code 4 questions for now
            if not forget_random:
                num_forget = int(4)
            # random then it depends on q_per_author and forgotten_fact_fraction
            elif forget_random:
                num_forget = int(q_per_author * forgotten_fact_fraction)
        # non stratified random: forgotten_fact_fraction applied across entire dataset
        elif not stratified:
            num_forget = int(num_authors * q_per_author * forgotten_fact_fraction)

    # Get full list of indices
    all_indices = list(range(all_data.num_rows))

    # Get full list of authors
    authors = list(range(num_authors))

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
                        author, range(q_per_author), q_per_author
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
                forget_authors, forget_questions, q_per_author
            )
            retain_indices = list(set(all_indices).difference(forget_indices))

    # if granularity == author we just use retain/forget authors to define our splits
    elif granularity == "author":
        retain_indices = get_forget_indices(
            retain_authors, forget_questions, q_per_author
        )
        forget_indices = get_forget_indices(
            forget_authors, forget_questions, q_per_author
        )

    # create datasets
    forget_set, retain_set = Dataset.from_dict(
        all_data[forget_indices]
    ), Dataset.from_dict(all_data[retain_indices])

    return forget_set, retain_set
