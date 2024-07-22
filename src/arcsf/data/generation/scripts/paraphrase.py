import json

from datasets import Dataset, DatasetDict
from tqdm import tqdm

from arcsf.data.generation.gpt_utils import (
    AnswerHallucinator,
    paraphrase_question_answer,
)

# Paraphrasing questions, answers, and their perturbed counterparts
with open("temp/gen_tofu/all_items.json", "r") as entity_file:
    entity_dict = json.load(entity_file)

entity_list = []
for key, item in entity_dict.items():
    entity_list.append({"key": key, "type": item["type"], "data": item["data"]})

with open("temp/gen_tofu/questions_with_paraphrase.json") as question_file:
    question_list = json.load(question_file)

answer_hallucinator = AnswerHallucinator()

for question_dict in tqdm(question_list, desc="Paraphrasing Questions"):
    paraphrased = paraphrase_question_answer(question_dict)
    question_dict["paraphrased_answer"] = paraphrased[1]
    placeholder_question = question_dict["question"]
    question_dict["question"] = paraphrased[0]
    hallucinated = answer_hallucinator.hallucinate_answer(question_dict)
    question_dict["question"] = placeholder_question
    question_dict["paraphrased_question"] = paraphrased[0]
    question_dict["paraphrased_hallucinated_answers"] = hallucinated


# GENERATING DATSET
# THESE NEED TO BE LISTS
def question_yielder():
    yield from question_list


def entity_yielder():
    yield from entity_list


question_dataset = Dataset.from_generator(question_yielder)
entity_dataset = Dataset.from_generator(entity_yielder)
full_dataset = DatasetDict(
    {"question_data": question_dataset, "entity_data": entity_dataset}
)
full_dataset.save_to_disk("temp/gen_tofu/new_dataset/")
print(full_dataset)
