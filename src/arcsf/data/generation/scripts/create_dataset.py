import argparse
import json

from datasets import Dataset


# GENERATOR FOR CREATING DATASET
def question_yielder(question_list):
    yield from question_list


def main(args):
    gen_tofu_path = args.gen_tofu_path

    with open(f"{gen_tofu_path}/questions.json") as question_file:
        question_list = json.load(question_file)

    full_dataset = Dataset.from_generator(question_yielder(question_list))
    full_dataset.save_to_disk(f"{gen_tofu_path}/dataset/")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "From a directory containing a question json, creates a huggingface dataset"
        )
    )
    parser.add_argument(
        "gen_tofu_path",
        type=str,
        help="Relative path to directory containing the data.",
    )
    args = parser.parse_args()
