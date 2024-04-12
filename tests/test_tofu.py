import numpy as np
from datasets import Dataset

from arcsf.data.tofu import load_tofu


def _check_type(dataset):
    # check datasets are in fact datasets
    assert isinstance(dataset, Dataset)


def _check_dropped_qs(granularity, stratify, forget_random, debug_dict):
    # check correct number of questions dropped per author/indices
    # lie in range of forget authors -> not needed if random

    forgot_q_frac = 0.1
    q_per_author, forget_authors = (
        debug_dict["q_per_author"],
        debug_dict["forget_author_numbers"],
    )
    forget_indices = np.array(debug_dict["forget_indices"])
    check_list = list()

    # get number of questions dropped per author depending on granularity
    if stratify and not forget_random:
        author_qs_dropped = debug_dict["num_forget"]
    elif granularity == "question" and stratify and forget_random:
        author_qs_dropped = int(q_per_author * forgot_q_frac)
    else:
        raise ValueError(
            "Non-stratified random question dropping \
                should not call _check_dropped_qs()"
        )
    for author in forget_authors:
        # get range for each author
        min_index = author * q_per_author
        max_index = (author + 1) * q_per_author
        # get indices from dropped questions which are within this range
        matching_indices = np.where(
            np.logical_and(forget_indices >= min_index, forget_indices < max_index)
        )
        # take number of questions dropped -> will raise index error if too few
        check_indices = forget_indices[matching_indices[0]][0:author_qs_dropped]
        check_list.append(check_indices)
    # create new list of all forget indices
    check_list = np.sort(np.concatenate(check_list))
    # check equal to original:
    # will raise assertion error if too many questions from an author
    assert np.array_equal(check_list, np.sort(forget_indices))


def _check_dataset_len(
    granularity, stratified, forget_random, debug_dict, forgot_a_frac, forgot_q_frac
):
    n_authors, q_per_author, n_forget = (
        debug_dict["author_count"],
        debug_dict["q_per_author"],
        debug_dict["num_forget"],
    )
    forget_indices, _ = debug_dict["forget_indices"], debug_dict["retain_indices"]
    # check correct number of questions dropped

    if granularity == "author" and stratified:
        # n_questions_per_author*n_authors_dropped
        assert len(forget_indices) == int(n_authors * n_forget * forgot_a_frac)
    elif granularity == "question" and forget_random and stratified:
        # n_questions_per_author*frac_q_dropped*n_authors_dropped
        assert len(forget_indices) == int(
            n_authors * q_per_author * forgot_a_frac * forgot_q_frac
        )
    elif granularity == "author" and forget_random and not stratified:
        # n_questions_per_author*n_authors*frac_q_dropped
        assert len(forget_indices) == int(n_authors * q_per_author * forgot_q_frac)


def _test_load_tofu(granularity, stratified, forget_random, seed=42):

    forgot_a_frac = 0.1
    forgot_q_frac = 0.1

    forget_set, retain_set, debug_dict = load_tofu(
        granularity=granularity,
        stratified=stratified,
        forget_random=forget_random,
        forgotten_author_fraction=forgot_a_frac,
        forgotten_fact_fraction=forgot_q_frac,
        random_seed=seed,
        debug=True,
    )

    _check_dataset_len(
        granularity, stratified, forget_random, debug_dict, forgot_a_frac, forgot_q_frac
    )

    _check_type(forget_set)
    _check_type(retain_set)

    # check correct number of questions dropped per author/indices
    # lie in range of forget authors -> not needed if random
    if not (not stratified and forget_random):
        _check_dropped_qs(granularity, stratified, forget_random, debug_dict)


def test_load_tofu_random():
    _test_load_tofu(granularity="question", stratified=False, forget_random=True)


def test_load_tofu_random_within_authors():
    _test_load_tofu(granularity="question", stratified=True, forget_random=True)


def test_load_tofu_author_level():
    _test_load_tofu(granularity="author", stratified=True, forget_random=False)


def test_load_tofu_structured_within_authors():
    _test_load_tofu(granularity="question", stratified=True, forget_random=False)
