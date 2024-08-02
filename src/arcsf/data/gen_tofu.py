import json
import math
import random

import datasets
import pandas as pd
from datasets import Dataset, load_from_disk

from arcsf.data.generation.utils import KeyChecker

GEN_TOFU_PATH = "data/gen_tofu"


def _load_gen_tofu_from_disk() -> Dataset:
    return load_from_disk(f"{GEN_TOFU_PATH}/dataset")


def _load_gen_tofu_granularity(
    granularity: str,
    forget_fraction: float,
    random_seed: int,
) -> tuple[Dataset, Dataset] | tuple[None, Dataset]:
    """
    Basic load function for the generated dataset.

    Args:
        granularity: What level of granularity to perform forgetting. Currently takes
        "question", "publisher", "author", or "book".
        forget_fraction: Fraction of data to be removed.
        random_seed: seed for random elements.

    Returns:
        tuple of datasets, the first index containing the forget_set, and the second the
        retain set.
    """

    question_dataset = _load_gen_tofu_from_disk()

    if forget_fraction == 0.0:
        return None, question_dataset

    with open(f"{GEN_TOFU_PATH}/all_items.json") as entity_file:
        all_entities = json.load(entity_file)

    if granularity == "question":
        n_question = question_dataset.shape[0]
        all_indices = list(range(n_question))
        forget_indices = random.sample(
            all_indices,
            k=math.floor(n_question * forget_fraction),
        )
        retain_indices = [index for index in all_indices if index not in forget_indices]
        forget_split = question_dataset.select(forget_indices)
        retain_split = question_dataset.select(retain_indices)
    if granularity != "question":
        entity_type_keys = []
        for entity_key, entity_data in all_entities.items():
            if entity_data["type"] == granularity:
                entity_type_keys.append(entity_key)

        random.seed(random_seed)
        forget_keys = random.sample(
            entity_type_keys, k=math.floor(len(entity_type_keys) * forget_fraction)
        )

        forget_split = question_dataset.filter(
            KeyChecker(forget_keys, find_forget=True)
        )
        retain_split = question_dataset.filter(
            KeyChecker(forget_keys, find_forget=False)
        )

    return forget_split, retain_split


def _load_gen_tofu_relationship(
    forget_fraction: float,
    retain_subset: bool,
    random_seed: int,
) -> tuple[Dataset, Dataset] | tuple[None, Dataset]:
    """
    Basic load function for the generated dataset.

    Args:
        granularity: What level of granularity to perform forgetting. Currently takes
        "question", "publisher", "author", or "book".
        forget_fraction: Fraction of data to be removed.
        random_seed: seed for random elements.

    Returns:
        tuple of datasets, the first index containing the forget_set, and the second the
        retain set.
    """

    # Load data from disk
    question_dataset = _load_gen_tofu_from_disk()

    # Full set
    if forget_fraction == 0.0:
        return None, question_dataset

    # Load relationships
    all_relationships = pd.read_csv(
        f"{GEN_TOFU_PATH}/all_connections.csv",
        header=None,
        names=["entity_1", "entity_2"],
    )

    # Sample relationships to remove
    num_relationship = len(all_relationships)
    random.seed(random_seed)
    to_remove = random.sample(
        range(num_relationship),
        k=math.floor(forget_fraction * num_relationship),
    )

    # Get list of keys in dataset and all indices
    keys_list = question_dataset["keys"]
    all_indices = list(range(question_dataset.shape[0]))

    # Get list of forget indices
    forget_indices = []
    for index in to_remove:
        rel = all_relationships.iloc[index]
        forget_indices = forget_indices + [
            i
            for i, keys in enumerate(keys_list)
            if rel["entity_1"] in keys and rel["entity_2"] in keys
        ]

    # Get unique forget indices, retain indices
    forget_indices = set(forget_indices)
    retain_indices = [index for index in all_indices if index not in forget_indices]

    # Get splits
    forget_split = question_dataset.select(forget_indices)
    retain_split = question_dataset.select(retain_indices)

    # If wishing to obtain only the retain subset
    if retain_subset:
        forget_relationships = [all_relationships.iloc[index] for index in to_remove]
        entity_1 = [rel["entity_1"] for rel in forget_relationships]
        entity_2 = [rel["entity_2"] for rel in forget_relationships]
        forget_entities = set(entity_1 + entity_2)
        retain_split = retain_split.filter(
            KeyChecker(forget_entities, find_forget=True)
        )

    # Return
    return forget_split, retain_split


def load_gen_tofu(
    type: str, **kwargs
) -> tuple[Dataset, Dataset] | tuple[None, Dataset]:
    """
    Basic load function for the generated dataset.

    Args:
        type: Whether to create forget sets by granularity or relationships. Used to
              selct a load function
        **kwargs: Passed to loading function

    Returns:
        tuple of datasets, the first index containing the forget_set, and the second the
        retain set. If forget set is empty, first index will be None.
    """

    # Select load fn based on type
    if type == "granularity":
        load_fn = _load_gen_tofu_granularity
    if type == "relationship":
        load_fn = _load_gen_tofu_relationship

    # Pass kwargs and return
    return load_fn(**kwargs)


class GenTofuPerturber:
    """
    Function for retrieving perturbed (erroneous) samples at evaluation time
    """

    def __init__(self, data: datasets.Dataset, n_perturbed: int) -> None:
        """
        Intialising the perturbing class

        Args:
            data: dataset from which the samples are being perturbed
            n_perturbed: number of perturbed samples the __call__ function should output
        """
        self.data = data
        self.n_perturbed = n_perturbed
        assert n_perturbed <= 3, (
            "Currently only functionality for" " 3 or less perturbed samples"
        )

    def __call__(self, idx: int) -> list[str]:
        """
        Args:
            idx: index from the dataset from which the sample is being perturbed

        Returns:
            list of strings representing perturbed samples
        """
        return self.data[idx]["paraphrased_perturbed_answers"][: self.n_perturbed]
