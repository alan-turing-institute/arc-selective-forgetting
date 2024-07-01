import math
import random

from datasets import Dataset, load_dataset

from src.arcsf.data.generation.utils import KeyChecker

TOFU_PATH = "temp/gen_tofu/dataset"


def load_gen_tofu(
    granularity: str,
    forget_fraction: float,
    random_seed: int,
) -> tuple[Dataset, Dataset]:

    full_dataset = load_dataset(TOFU_PATH)
    question_dataset = full_dataset["question_data"]

    entity_type_keys = full_dataset["entity_data"].filter(
        lambda row: row["type"] == granularity
    )["key"]

    random.seed(random_seed)
    forget_keys = random.sample(
        entity_type_keys, k=math.floor(len(entity_type_keys) * forget_fraction)
    )

    forget_split = question_dataset.filter(KeyChecker(forget_keys, find_forget=True))
    retain_split = question_dataset.filter(KeyChecker(forget_keys, find_forget=False))

    return forget_split, retain_split
