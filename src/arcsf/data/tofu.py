import math

import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


def _flatten_list_of_lists(list: list[list]) -> list:
    return sum(list, [])


# returns the question index in the dataset, given an author number and question
# number (within the author) (inner function)
def _get_forget_index(
    author: int,
    question: int,
    q_per_author: int,
) -> int:
    return question + q_per_author * author


# As above returns the question indices in the dataset, given an
# author number and question numbers (within the author) (inner function)
def _get_forget_indices(
    authors: int,
    questions: int | list[int],
    q_per_author: int,
) -> int | list[int]:
    if isinstance(questions, int):
        return _get_forget_index(authors, questions, q_per_author)
    if isinstance(questions, list):
        return [
            _get_forget_index(authors, question, q_per_author) for question in questions
        ]


# As above returns the question indices in the dataset, given
# author numbers and question numbers (across dataset)
def get_forget_indices(
    authors: int | list[int],
    questions: int | list[int],
    q_per_author: int,
) -> list[int]:
    if isinstance(authors, int):
        return _get_forget_indices(authors, questions, q_per_author)
    if isinstance(authors, list):
        out = [
            _get_forget_indices(author, questions, q_per_author) for author in authors
        ]
        if isinstance(out[0], list):
            return _flatten_list_of_lists(out)
        return out


def load_tofu(
    granularity: str,
    stratified: bool,
    forget_random: bool,
    forgotten_author_fraction: float,
    forgotten_fact_fraction: float,
    random_seed: int,
    debug=False,
) -> tuple[Dataset, Dataset, dict] | tuple[Dataset, Dataset]:
    """
    Loads TOFU dataset given different flags for retain--forget split.
    Params:
        - granularity: level at which forgetting takes place (author vs question)
        - stratified: if forgetting questions restrain to specific authors?
        - forget_random: is forgetting happening randomly within constraints?
        - forgotten_author_fraction: fraction of authors from which to forget questions
        - forgotten_fact_fraction: fraction of questions to randomly forget
            - if stratified == True represents fraction of Qs in author
            - if stratified == False represents fraction of total Qs
        - random_seed: seed for reproducibility
        - debug: returns dictionary containing meta_data if being used during testing
    Returns:
        - Two datasets with forget and retain sets
            as well as a debugging dictionary (optional)
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

    debug_dict = {
        "author_count": num_authors,
        "q_per_author": q_per_author,
        "num_forget": num_forget,
    }

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

    debug_dict["forget_author_numbers"] = forget_authors

    if granularity == "question":
        if forget_random:
            if stratified:
                # if stratified and random: select train-test split within each
                # set of author questions (currently selects same indices
                # TODO: Is this the behaviour we want?
                forget_indices = []
                for author in forget_authors:
                    all_author_indices = np.arange(
                        get_forget_indices(author, 0, q_per_author),
                        get_forget_indices(author, 20, q_per_author),
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
                # need to redefine the forgotten authors now - can use np.floor
                debug_dict["forget_author_numbers"] = np.floor(
                    np.array(forget_indices) / 20
                ).tolist()
        # if not random then take first num_forget questions
        # forget_questions contains the relevant indices for this
        if not forget_random:
            forget_indices = get_forget_indices(
                forget_authors, forget_questions, q_per_author
            )
            retain_indices = list(set(all_indices).difference(forget_indices))

    # if granularity == author we just use retain/forget authors to define our splits
    if granularity == "author":
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

    # return datasets
    if debug:
        debug_dict["forget_indices"] = forget_indices
        debug_dict["retain_indices"] = retain_indices
        return forget_set, retain_set, debug_dict
    else:
        return forget_set, retain_set
