import argparse
import json

import datasets
from tqdm import tqdm

from arcsf.data.generation.utils import KeyChecker
from arcsf.utils import hf_progress_bars_disabled


def main(args):
    data_path = args.data_path
    # LOAD DATASET OBJECT
    question_dataset = datasets.load_from_disk(f"{data_path}/dataset")

    with open(f"{data_path}/all_items.json", "r") as item_file:
        all_items = json.load(item_file)

    # GENERATING SOME SPLITS FROM THE DICTIONARY OBJECT
    authors = []
    publishers = []
    books = []

    for key, item in all_items.items():
        if item["type"] == "author":
            authors.append(key)
        elif item["type"] == "publisher":
            publishers.append(key)
        elif item["type"] == "book":
            books.append(key)

    types = {"books": [], "authors": [], "publishers": []}
    for entity_index, entity_type in enumerate([books, authors, publishers]):
        entity_name = list(types.keys())[entity_index]
        for entity_key in tqdm(entity_type, desc=entity_name):
            # PUBLISHER SPLIT
            with hf_progress_bars_disabled():
                forget_split = question_dataset.filter(
                    KeyChecker(entity_key, find_forget=True)
                )
            types[entity_name].append(len(forget_split))

    with open("temp/split_analysis.json", "w") as item_file:
        json.dump(types, item_file, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=("From a directory containing the data, generates some test splits")
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Relative path to directory containing the data.",
    )

    args = parser.parse_args()
    main(args)
