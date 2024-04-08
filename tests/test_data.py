import numpy as np

from arcsf.utils.data_utils import load_tofu


def _test_load_tofu(granularity, seed=42):

    forgot_a_frac = 0.1
    forgot_q_frac = 0.1

    _, _, debug_dict = load_tofu(
        granularity=granularity,
        forgotten_author_fraction=forgot_a_frac,
        forgotten_fact_fraction=forgot_q_frac,
        random_seed=seed,
        debug=True,
    )

    # meta data
    n_authors, q_per_author, forget_authors = (
        debug_dict["author_count"],
        debug_dict["author_q_count"],
        debug_dict["forget_author_numbers"],
    )
    # selected questions
    forget_indices, _ = debug_dict["forget_indices"], debug_dict["retain_indices"]

    # check correct number of questions dropped
    if granularity == "author_level":
        assert len(forget_indices) == int(n_authors * q_per_author * forgot_a_frac)
    elif granularity == "random_within_authors":
        assert len(forget_indices) == int(
            n_authors * q_per_author * forgot_a_frac * forgot_q_frac
        )
    elif granularity == "structured_within_authors":
        assert len(forget_indices) == len(forget_authors) * debug_dict["qs_to_remove"]
    elif granularity == "random":
        assert len(forget_indices) == int(n_authors * q_per_author * forgot_q_frac)

    # check correct number of questions dropped per author/indices
    # lie in range of forget authors -> not needed if random
    if granularity != "random":
        check_list = list()

        # get number of questions dropped per author depending on granularity
        if granularity == "structured_within_authors":
            author_qs_dropped = debug_dict["qs_to_remove"]
        elif granularity == "random_within_authors":
            author_qs_dropped = int(q_per_author * forgot_q_frac)
        elif granularity == "author_level":
            author_qs_dropped = q_per_author

        for author in forget_authors:
            # get range for each author
            min_index = author * q_per_author
            max_index = (author + 1) * q_per_author
            # get indices from dropped questions which are within this range
            matching_indices = np.where(
                np.logical_and(forget_indices >= min_index, forget_indices < max_index)
            )
            # take number of questions dropped -> will raise index error if too few
            check_indices = forget_indices[matching_indices][0:author_qs_dropped]
            check_list.append(check_indices)
        # create new list of all forget indices
        check_list = np.sort(np.concatenate(check_list))
        # check equal to original:
        # will raise assertion error if too many questions from an author
        assert np.array_equal(check_list, forget_indices)


def test_load_tofu():
    options = [
        "random",
        "random_within_authors",
        "author_level",
        "structured_within_authors",
    ]
    for granularity in options:
        _test_load_tofu(granularity=granularity)
