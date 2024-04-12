import numpy as np
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


def flatten_list_of_lists(list: list[list]) -> list:
    return sum(list, [])


def _get_forget_index(
    author: int,
    question: int,
    q_per_author: int,
) -> int:
    return question + q_per_author * author


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
            return flatten_list_of_lists(out)
        return out


def load_tofu(
    granularity: str,
    stratified: bool,
    forget_random: bool,
    forgotten_author_fraction: float,
    forgotten_fact_fraction: float,
    random_seed: int,
    debug=False,
) -> tuple[Dataset, Dataset, dict]:

    all_data = load_dataset("locuslab/TOFU", "full")["train"]

    num_authors = 200  # hard coding author count for now
    q_per_author = 20  # hard coding author question count for now
    if granularity == "author":
        num_forget = q_per_author
    elif granularity == "question":
        if stratified:
            if not forget_random:
                num_forget = int(4)
            elif forget_random:
                num_forget = int(q_per_author * forgotten_fact_fraction)
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
        questions = list(range(num_forget))

    retain_authors, forget_authors = train_test_split(
        authors,
        test_size=forgotten_author_fraction,
        random_state=random_seed,
    )

    debug_dict["forget_author_numbers"] = forget_authors

    # Get indices
    if granularity == "question":
        if forget_random:
            if stratified:
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
                forget_indices = np.concatenate(forget_indices)
                retain_indices = np.array(
                    list(set(all_indices).difference(forget_indices))
                )
            else:
                retain_indices, forget_indices = train_test_split(
                    all_indices,
                    test_size=num_forget,
                    random_state=random_seed,
                )
        if not forget_random:
            forget_indices = get_forget_indices(forget_authors, questions, q_per_author)
            retain_indices = list(set(all_indices).difference(forget_indices))

    if granularity == "author":
        retain_indices = get_forget_indices(retain_authors, questions, q_per_author)
        forget_indices = get_forget_indices(forget_authors, questions, q_per_author)

    forget_set, retain_set = Dataset.from_dict(
        all_data[forget_indices]
    ), Dataset.from_dict(all_data[retain_indices])

    if debug:
        debug_dict["forget_indices"] = forget_indices
        debug_dict["retain_indices"] = retain_indices
        return forget_set, retain_set, debug_dict
    else:
        return forget_set, retain_set
