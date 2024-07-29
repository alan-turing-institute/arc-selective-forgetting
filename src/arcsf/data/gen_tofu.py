import json
import math
import random

from datasets import Dataset, load_from_disk

from arcsf.data.generation.utils import KeyChecker

GEN_TOFU_PATH = "temp/gen_tofu"


def load_gen_tofu(
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

    question_dataset = load_from_disk(f"{GEN_TOFU_PATH}/dataset")

    if forget_fraction == 0.0:
        return None, question_dataset

    with open(f"{GEN_TOFU_PATH}/all_items.json") as entity_file:
        all_entities = json.load(entity_file)

    if granularity == "question":
        n_question = question_dataset.shape[0]
        forget_keys = random.sample(
            list(range(n_question)),
            k=math.floor(n_question * forget_fraction),
        )
    if granularity != "question":
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

    def __init__(self, data, n_perturbed):
        self.data = data
        self.n_perturbed = n_perturbed
        assert n_perturbed <= 3, (
            "Currently only functionality for" " 3 or less perturbed samples"
        )

    def __call__(self, idx):
        return self.data[idx]["paraphrased_perturbed_answers"][: self.n_perturbed]
