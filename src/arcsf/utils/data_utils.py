import itertools

import numpy as np
from datasets import Dataset, load_dataset


def get_indices_structured(
    forgotten_author_numbers: list,
    q_remove: int = 4,
    n_authors: int = 200,
    q_per_author: int = 20,
):
    """
    Returns two flattened lists of Question--Answer indices given authors numbers to
    remove/retain, in the case where the forget target can be easily structured.
    Assumes equal questions per author, and sorted authors.

        Parameters:
                forgotten_author_numbers (list):    List of integers containing author
                                                    indexes about which facts should
                                                    be removed
                n_authors (int):                    Total number of authors
                q_removal (int):                    Number of questions/facts that
                                                    should be removed
                q_per_author (int):                 Integer defining the number of
                                                    questions to remove per author

        Returns:
                forget_indices (list):              List of integers corresponding to
                                                    the question indices pertaining the
                                                    authors in author_numbers
                retain_indices (list):              List of integers corresponding to
                                                    the remaining question indices
    """
    assert (
        q_remove <= q_per_author
    ), "Cannot remove more questions for an author than there are questions per author."
    indices_list = list()
    for author_n in forgotten_author_numbers:
        ref_index = author_n * q_per_author  # start index for author questions
        indices_list.append(np.arange(ref_index, ref_index + q_remove))
    forget_indices = np.sort(
        np.array(list(itertools.chain.from_iterable(indices_list)))
    )
    retain_indices = np.delete(np.arange(0, n_authors * q_per_author), forget_indices)
    return forget_indices, retain_indices


def get_indices_random(
    within_author: bool = True,
    forgotten_author_numbers: list = None,
    q_remove_fraction: int = 0.1,
    n_authors: int = 200,
    q_per_author: int = 20,
    random_seed: int = 42,
):
    """
    Returns two flattened lists of Question--Answer indices given authors numbers
    to remove/retain. In the case where the forget target can be easily structured.
    Assumes equal questions per author, and sorted authors.

        Parameters:
                within_author (bool):               Boolean variable specifying whether
                                                    the random dropping occurs within
                                                    specific authors
                forgotten_author_numbers (list):    List of integers containing author
                                                    indexes about which facts should
                                                    be removed
                q_remove_fraction (float):          Fraction of questions/facts that
                                                    should be removed
                                                        - If within_author = True:
                                                            This is the fraction
                                                            of q_per_author
                                                        - If within_author = False:
                                                            This is the fraction of
                                                            total number of questions
                                                            (n_questions)
                n_authors (int):                    Total number of authors
                q_per_author (int):                 Integer denoting the number of
                                                    questions to remove per author
                random_seed (int):                  Seed for random properties

        Returns:
                forget_indices (list):              List of integers corresponding to
                                                    the question indices to be forgotten
                retain_indices (list):              List of integers corresponding to
                                                    the remaining question indices
    """
    rng = np.random.default_rng(random_seed)
    n_questions = n_authors * q_per_author
    if within_author:
        assert (
            forgotten_author_numbers is not None
        ), "If removing questions within specific authors, specify authors."
        assert (
            int(q_remove_fraction * q_per_author) >= 1
        ), "At least 1 question needs to be removed per author."
        indices_list = list()
        for author_n in forgotten_author_numbers:
            ref_index = author_n * q_per_author  # start index for author questions
            indices_list.append(
                rng.choice(
                    np.arange(ref_index, ref_index + q_per_author),
                    int(q_per_author * q_remove_fraction),
                    replace=False,
                )
            )
        forget_indices = np.sort(
            np.array(list(itertools.chain.from_iterable(indices_list)))
        )

    else:
        forget_indices = np.sort(
            rng.choice(
                np.arange(0, n_questions),
                int(n_questions * q_remove_fraction),
                replace=False,
            )
        )

    retain_indices = np.delete(np.arange(0, n_questions), forget_indices)
    return forget_indices, retain_indices


def get_author_splits(
    n_authors: int = 200, author_forget_fraction: float = 0.1, random_seed: int = 42
):
    """
    Returns randomly selected author numbers to retain/forget.

        Parameters:
                n_authors (list):       Total number of authors
                forget_fraction (float):Fraction of authors that should be removed
                random_seed (int):      Random seed for reproducibility

        Returns:
                forget_authors (list):  List of integers containing author indexes to
                                        remove/forget
    """
    rng = np.random.default_rng(random_seed)
    author_options = np.arange(0, n_authors)
    forget_authors = np.sort(
        rng.choice(
            author_options, int(author_forget_fraction * n_authors), replace=False
        )
    )

    return forget_authors


def load_tofu(
    granularity: str = "author_level",
    forgotten_author_fraction: float = 0.1,
    forgotten_fact_fraction: float = 0.1,
    random_seed: int = 42,
    debug=False,
):
    """
    Loads the TOFU dataset with randomly chosen authors/information to forget,
    using question indices to remove/retain data.

        Parameters:
                granularity (str):                          Granularity at which to
                                                            perform unlearning
                forgotten_author_fraction (float):          Fraction of authors that
                                                            should be removed
                forgotten_fact_fraction (float):            Fraction of facts to be
                                                            removed (currently not used)
                random_seed (int):                          Random seed for
                                                            reproducibility

        Returns:
                forget_set (datasets.arrow_dataset.Dataset):    Dataset containing
                                                                removed/forgotten
                                                                authors
                retain_set (datasets.arrow_dataset.Dataset):    Dataset containing
                                                                retained authors
                debug_dict (dict):                              Dictionary containing
                                                                information for
                                                                debugging, used in tests
    """

    all_data = load_dataset("locuslab/TOFU", "full")[
        "train"
    ]  # load all data to work with
    author_count = 200  # hard coding author count for now
    author_q_count = 20  # hard coding author question count for now
    qs_to_remove = 4
    debug_dict = {"author_count": author_count, "author_q_count": author_q_count}

    forget_author_numbers = get_author_splits(
        author_count, forgotten_author_fraction, random_seed
    )
    debug_dict["forget_author_numbers"] = forget_author_numbers

    if granularity == "author_level":
        forget_indices, retain_indices = get_indices_structured(
            forget_author_numbers,
            q_remove=author_q_count,
            n_authors=author_count,
            q_per_author=author_q_count,
        )

    elif granularity == "structured_within_authors":
        # biographical information generally contained in first 4 questions
        forget_indices, retain_indices = get_indices_structured(
            forget_author_numbers,
            q_remove=qs_to_remove,
            n_authors=author_count,
            q_per_author=author_q_count,
        )
        debug_dict["qs_to_remove"] = qs_to_remove

    elif granularity == "random_within_authors":
        forget_indices, retain_indices = get_indices_random(
            True,
            forgotten_author_numbers=forget_author_numbers,
            q_remove_fraction=forgotten_fact_fraction,
            n_authors=author_count,
            q_per_author=author_q_count,
            random_seed=random_seed,
        )

    elif granularity == "random":
        # biographical information generally contained in first 4 questions
        forget_indices, retain_indices = get_indices_random(
            False,
            forgotten_author_numbers=None,
            q_remove_fraction=forgotten_fact_fraction,
            n_authors=author_count,
            q_per_author=author_q_count,
            random_seed=random_seed,
        )

    forget_set, retain_set = Dataset.from_dict(
        all_data[forget_indices]
    ), Dataset.from_dict(all_data[retain_indices])

    if debug:
        debug_dict["forget_indices"] = forget_indices
        debug_dict["retain_indices"] = retain_indices
        return forget_set, retain_set, debug_dict
    else:
        return forget_set, retain_set
