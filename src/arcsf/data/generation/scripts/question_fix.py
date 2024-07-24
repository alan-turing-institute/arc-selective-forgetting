import json

from arcsf.data.generation.gpt_generation import IterativeGenerator

# file for fixing problem profiles eg. names overlap with real books

target_keys = [
    "15f3df38-c14e-45e4-8f53-d1962b4e48b3",
    "75085105-29e0-446c-94e3-6cccbdbb1744",
    "acf5eb70-7532-4902-ac7c-dfd8de95d6d9",
    "2073777f-a066-43c5-be7d-f4098ebfdef5",
]

name_switch_dict = {
    "Fatal Trust": "Collateral Trust",
    "In Cold Blood": "Cold Blooded",
    "The Woman in Black": "The Haunting of the Castle on the Hill",
    "Silent Partner": "Pills of Death",
}

with open("temp/gen_tofu_full/questions.json", "r") as question_file:
    questions = json.load(question_file)

print(len(questions))

with open("temp/gen_tofu_full/all_items.json", "r") as entity_file:
    entity_dict = json.load(entity_file)

for target_key in target_keys:
    entity_dict[target_key]["data"]["name"] = name_switch_dict[
        entity_dict[target_key]["data"]["name"]
    ]

question_indices = []
new_questions = []
for question_index, question_dict in enumerate(questions):
    for query_key in question_dict["keys"]:
        if query_key in target_keys:
            new_question = question_dict
            for target_name, replacement_name in name_switch_dict.items():
                new_question["answer"] = new_question["answer"].replace(
                    target_name, replacement_name
                )
                new_question["question"] = new_question["question"].replace(
                    target_name, replacement_name
                )
                new_question["keys"] = new_question["keys"]
            new_questions.append(new_question)
            question_indices.append(question_index)

for index in sorted(question_indices, reverse=True):
    del questions[index]

print(len(questions))

# Cut off generated questions (fortunately these are all at the end)
new_questions = new_questions[:24]

iterative_generator = IterativeGenerator(entity_dict)
n_gen = 2
for target_key in target_keys:
    book_profile = entity_dict[target_key]["data"]
    keys = [
        book_profile["key"],
        book_profile["genre"],
        book_profile["author"],
        book_profile["length"],
        book_profile["sales"],
        book_profile["awards"],
        book_profile["publisher"],
    ]
    qa_pairs = []
    for key_index in [2, 4, 8]:
        iteration_keys = keys[:key_index]
        iteration_qa_pairs = iterative_generator.iterate_book_questions(
            book_profile, n_gen + 1, qa_pairs, iteration_keys
        )
        for iteration_qa in iteration_qa_pairs[: n_gen + 1]:
            iteration_row = {
                "question": iteration_qa[0],
                "answer": iteration_qa[1],
                "keys": iteration_keys,
            }
            qa_pairs.append(iteration_row)

    for qa in qa_pairs:
        new_questions.append(qa)

for new_question_dict in new_questions:
    print("\n")
    print(new_question_dict["question"])
    print(new_question_dict["answer"])
    print(new_question_dict["keys"])
    questions.append(new_question_dict)

print(len(questions))

with open("temp/gen_tofu_full/fixed_all_items.json", "w") as entity_file:
    json.dump(entity_dict, entity_file, indent=2)


with open("temp/gen_tofu_full/fixed_questions.json", "w") as question_file:
    json.dump(questions, question_file, indent=2)
