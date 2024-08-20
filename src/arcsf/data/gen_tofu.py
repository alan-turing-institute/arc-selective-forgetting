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
    **kwargs,
) -> tuple[Dataset, Dataset] | tuple[None, Dataset]:
    """
    Load function for generated dataset. Creates forget split by granularity.

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

    random.seed(random_seed)

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

    else:
        entity_type_keys = []
        for entity_key, entity_data in all_entities.items():
            if entity_data["type"] == granularity:
                entity_type_keys.append(entity_key)

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
    random_seed: int,
    retain_subset: bool = False,
    find_forget: bool = True,
    **kwargs,
) -> tuple[Dataset, Dataset] | tuple[None, Dataset]:
    """
    Load function for generated dataset. Creates forget split by relationships.

    Args:
        forget_fraction: Fraction of data to be removed.
        random_seed: seed for random elements.
        retain_subset: If True, return a subset of the retain split
        find_forget: If True, return only questions in the retain set containing
                     entities which have a relationship in the forget set. Only used if
                     retain_subset is True

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
            KeyChecker(forget_entities, find_forget=find_forget)
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

    def __init__(
        self, data: datasets.Dataset, n_perturbed: int, perturbed_key: str
    ) -> None:
        """
        Intialising the perturbing class

        Args:
            data: dataset from which the samples are being perturbed
            n_perturbed: number of perturbed samples the __call__ function should output
            answer_key: dictionary_key used to identify the question
        """
        self.data = data
        self.perturbed_key = perturbed_key
        self.n_perturbed = n_perturbed
        if n_perturbed > 3:
            raise ValueError(
                "Currently only functionality for 3 or less perturbed samples"
            )

    def __call__(self, idx: int) -> list[str]:
        """
        Args:
            idx: index from the dataset from which the sample is being perturbed

        Returns:
            list of strings representing perturbed samples
        """
        return self.data[idx][self.perturbed_key][: self.n_perturbed]
