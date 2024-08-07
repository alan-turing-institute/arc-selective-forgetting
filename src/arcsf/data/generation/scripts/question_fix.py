import json

from arcsf.data.generation.gpt_generation import IterativeGenerator

# file for fixing problem profiles eg. names overlap with real books

REGEN = False

target_keys = []

name_switch_dict = {
    "Riley Press": "Riley House Publising",
    "Catalyst Press": "Inhibition Press",
    "Quantum Publishing": "Atomic Press",
    "Silver Leaf Books": "Copper Leaf Books",
    "Sapphire Ink": "Ruby Ink Publishing",
    "Legacy Press": "Patrimony Press",
    "Crestwood Publishers": "House of Wood Press",
    "Pinnacle Publishing": "Meridian House Publishing",
    "Radiant Reads": "House of Order Press",
    "Celestial Publishing": "Astel Publishing",
    "Paragon Press": "Renegade Books",
    "Starlit Press": "Lightside House Publishing",
    "Midnight Ink": "Lunis Press",
}


with open("data/gen_tofu/questions.json", "r") as question_file:
    questions = json.load(question_file)

print(len(questions))

with open("data/gen_tofu/all_items.json", "r") as entity_file:
    entity_dict = json.load(entity_file)

for key, item in entity_dict.items():
    name = item["data"]["name"]
    if name in list(name_switch_dict.keys()):
        entity_dict[key]["data"]["name"] = name_switch_dict[name]
        target_keys.append(key)

capital_enumerations = {}
for key_name, switch_name in name_switch_dict.items():
    capital_enumerations[key_name.lower()] = switch_name
    capital_enumerations[key_name.capitalize()] = switch_name

name_switch_dict.update(capital_enumerations)

question_indices = []
new_questions = []
for question_index, question_dict in enumerate(questions):
    for query_key in question_dict["keys"]:
        new_question = question_dict
        if query_key in target_keys:
            for question_prop in [
                "question",
                "answer",
                "perturbed_answers",
                "paraphrased_question",
                "paraphrased_answer",
                "paraphrased_perturbed_answers",
            ]:

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

print(len(questions))

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
