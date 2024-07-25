import json

from datasets import Dataset

GEN_TOFU_PATH = "temp/gen_tofu/"

with open(f"{GEN_TOFU_PATH}/questions.json") as question_file:
    question_list = json.load(question_file)


# GENERATOR FOR CREATING DATASET
def question_yielder():
    yield from question_list


full_dataset = Dataset.from_generator(question_yielder)
full_dataset.save_to_disk(f"{GEN_TOFU_PATH}/dataset/")
