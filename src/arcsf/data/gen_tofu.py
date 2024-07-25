import json
import math
import random

import datasets
from datasets import Dataset, load_from_disk

from arcsf.data.generation.utils import KeyChecker

GEN_TOFU_PATH = "temp/gen_tofu"


def load_gen_tofu(
    granularity: str,
    forget_fraction: float,
    random_seed: int,
) -> tuple[Dataset, Dataset]:
    """
    Basic load function for the generated dataset.

    Args:
        granularity: What level of granularity to perform forgetting. Currently takes
        "publisher", "author", or "book".
        forget_fraction: Fraction of data to be removed.
        random_seed: seed for random elements.

    Returns:
        tuple of datasets, the first index containing the forget_set, and the second the
        retain set.
    """

    question_dataset = load_from_disk(f"{GEN_TOFU_PATH}/dataset")

    with open(f"{GEN_TOFU_PATH}/all_items.json") as entity_file:
        all_entities = json.load(entity_file)

    entity_type_keys = []
    for entity_key, entity_data in all_entities.items():
        if entity_data["type"] == granularity:
            entity_type_keys.append(entity_key)

    random.seed(random_seed)
    forget_keys = random.sample(
        entity_type_keys, k=math.floor(len(entity_type_keys) * forget_fraction)
    )

    forget_split = question_dataset.filter(KeyChecker(forget_keys, find_forget=True))
    retain_split = question_dataset.filter(KeyChecker(forget_keys, find_forget=False))

    return forget_split, retain_split


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
