import json

from arcsf.data.generation.gpt_generation import IterativeGenerator

# file for fixing problem profiles eg. names overlap with real books

REGEN = False

target_keys = ["33e47b8c-72fc-4780-8051-607065e28b08"]

name_switch_dict = {
    "Pelican Publishing": "Fixed Flock Press",
    "Pelican publishing": "Fixed Flock Press",
    "pelican publishing": "Fixed Flock Press",
}

with open("data/gen_tofu/questions.json", "r") as question_file:
    questions = json.load(question_file)

print(len(questions))

with open("data/gen_tofu/all_items.json", "r") as entity_file:
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
            new_question = {}
            for question_prop in [
                "question",
                "answer",
                "perturbed_answers",
                "paraphrased_question",
                "paraphrased_answer",
                "paraphrased_perturbed_answers",
            ]:
                new_question[question_prop] = question_dict[question_prop]
                for target_name, replacement_name in name_switch_dict.items():
                    if isinstance(new_question[question_prop], list):
                        for new_q_index, new_q in enumerate(
                            new_question[question_prop]
                        ):
                            new_question[question_prop][new_q_index] = new_q.replace(
                                target_name, replacement_name
                            )
                    else:
                        new_question[question_prop] = new_question[
                            question_prop
                        ].replace(target_name, replacement_name)

            new_question["keys"] = question_dict["keys"]
            new_questions.append(new_question)
            question_indices.append(question_index)
            continue

for question_index, new_question in zip(question_indices, new_questions):
    questions[question_index] = new_question

# Cut off generated questions (fortunately these are all at the end)
if REGEN:

    for index in sorted(question_indices, reverse=True):
        del questions[index]
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

with open("data/gen_tofu/fixed_all_items.json", "w") as entity_file:
    json.dump(entity_dict, entity_file, indent=2)


with open("data/gen_tofu/fixed_questions.json", "w") as question_file:
    json.dump(questions, question_file, indent=2)
