import json

from tqdm import tqdm

from arcsf.data.generation.gpt_generation import AnswerHallucinator

with open("temp/gen_tofu_full/questions_with_hallucinated.json") as question_file:
    question_list = json.load(question_file)

answer_hallucinator = AnswerHallucinator()
repeat_indices = []
failed_indices = []

for q_index in tqdm(repeat_indices):
    question_dict = question_list[q_index]
    try:
        hallucinated_answers = answer_hallucinator.hallucinate_answer(question_dict)
        question_dict["perturbed_answers"] = hallucinated_answers
        question_list[q_index] = question_dict
    except AttributeError:
        failed_indices.append(q_index)
        print(f"{q_index} failed")
        continue

print(failed_indices)

with open("temp/gen_tofu_full/questions_with_hallucinated_full.json", "w") as item_file:
    json.dump(question_list, item_file, indent=2)
