import json

from tqdm import tqdm

from arcsf.data.generation.gpt_generation import (
    AnswerHallucinator,
    paraphrase_question_answer,
)

# Paraphrasing questions, answers, and their perturbed counterparts
# with open("temp/gen_tofu/all_items.json", "r") as entity_file:
#     entity_dict = json.load(entity_file)

# entity_list = []
# for key, item in entity_dict.items():
#     entity_list.append({"key": key, "type": item["type"], "data": item["data"]})

with open("temp/gen_tofu_full/questions_with_paraphrase_full.json") as question_file:
    question_list = json.load(question_file)

answer_hallucinator = AnswerHallucinator()
repeat_indices = [732, 954, 1642, 2365, 2540, 3421, 6340, 6363, 6648, 6925, 9226]
failed_indices = []

for q_index in tqdm(repeat_indices):
    question_dict = question_list[q_index]
    # Create a new dictionary item and swap the keys
    try:
        new_dict = {}
        new_dict["keys"] = question_dict["keys"]
        new_dict["paraphrased_answer"] = question_dict["answer"]
        new_dict["paraphrased_question"] = question_dict["question"]
        new_dict["paraphrased_perturbed_answers"] = question_dict["perturbed_answers"]
        # Paraphrase the question answer pair
        paraphrased = paraphrase_question_answer(question_dict)
        new_dict["question"] = paraphrased[0]
        new_dict["answer"] = paraphrased[1]
        # hallucinate responses to the paraphrased answers
        hallucinated = answer_hallucinator.hallucinate_answer(new_dict)
        new_dict["perturbed_answers"] = hallucinated
        question_list[q_index] = new_dict
    except AttributeError:
        failed_indices.append(q_index)
        print(f"{q_index} failed")
        print(question_dict["question"])
        print(question_dict["answer"])
        continue

print(failed_indices)

with open("temp/gen_tofu_full/questions_with_paraphrase_full.json", "w") as item_file:
    json.dump(question_list, item_file, indent=2)

# GENERATING DATSET
# THESE NEED TO BE LISTS
# def question_yielder():
#     yield from question_list


# def entity_yielder():
#     yield from entity_list


# question_dataset = Dataset.from_generator(question_yielder)
# entity_dataset = Dataset.from_generator(entity_yielder)
# full_dataset = DatasetDict(
#     {"question_data": question_dataset, "entity_data": entity_dataset}
# )
# full_dataset.save_to_disk("temp/gen_tofu/paraphrased_dataset/")
# print(full_dataset)
