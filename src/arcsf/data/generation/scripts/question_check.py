import argparse
import json
import random
import re
import string

from tqdm import tqdm

from arcsf.data.generation.gpt_generation import (
    AnswerHallucinator,
    paraphrase_question_answer,
)


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value.lower() != val.lower()]


def get_hallucinated(question_matches, answer_matches):
    for question_match in question_matches:
        answer_matches = remove_values_from_list(answer_matches, question_match)
    return answer_matches


def main(args):
    data_path = args.data_path
    regen = args.regen
    verbose = args.verbose
    name_check = args.name_check
    perturbed_fix = args.perturbed_fix

    with open(f"{data_path}/questions_perturbed_fix.json") as question_file:
        question_list = json.load(question_file)

    with open(f"{data_path}/all_items.json") as entity_file:
        all_items = json.load(entity_file)

    signal_strings = [
        "rephrase",
        "restate",
        "paraphrase",
        "reformulate",
        "question and answer",
        "Q&A",
        "QA",
    ]

    problem_indices = []
    problems = []
    for q_index in tqdm(range(len(question_list))):
        question_dict = question_list[q_index]
        question_problem_matches = re.findall(
            r"|".join(signal_strings), question_dict["question"], re.IGNORECASE
        )
        answer_problem_matches = re.findall(
            r"|".join(signal_strings), question_dict["answer"], re.IGNORECASE
        )
        if len(question_problem_matches) > 0:
            problem_indices.append(q_index)
            problems.append(["Question:"] + question_problem_matches)
            continue
        if len(answer_problem_matches) > 0:
            problem_indices.append(q_index)
            problems.append(["Answer:"] + answer_problem_matches)
            continue
        if name_check:
            names = [
                all_items[entity_key]["data"]["name"]
                for entity_key in question_dict["keys"]
            ]
            regex = r"|".join(names)
            question_matches = re.findall(
                regex, question_dict["question"], re.IGNORECASE
            )
            answer_matches = re.findall(regex, question_dict["answer"], re.IGNORECASE)
            whole_names = [
                rf"\b{all_items[entity_key]['data']['name']}\b"
                for entity_key in question_dict["keys"]
            ]
            perturbed_regex = "|".join(whole_names)
            perturbed_answers = question_dict["paraphrased_perturbed_answers"]
            perturbed_problem_matches = []
            for perturbed_answer in perturbed_answers:
                perturbed_question_matches = re.findall(
                    perturbed_regex,
                    question_dict["paraphrased_question"],
                    re.IGNORECASE,
                )
                perturbed_answer_matches = re.findall(
                    perturbed_regex, perturbed_answer, re.IGNORECASE
                )
                perturbed_problem_matches += get_hallucinated(
                    perturbed_question_matches, perturbed_answer_matches
                )
            if len(question_matches) < 1:
                problem_indices.append(q_index)
                problems.append(["Question:"] + question_matches)
                continue
            if len(answer_matches) < 1:
                problem_indices.append(q_index)
                problems.append(["Answer:"] + answer_matches)
                continue
            if len(perturbed_problem_matches) > 0:
                problem_indices.append(q_index)
                problems.append(["Perturbed Answers:"] + perturbed_problem_matches)
                continue

    print(
        f"Identified  {len(problem_indices)} rows with issues.\n"
        f"(~{round((len(problem_indices)/len(question_list))*100, 1)}%"
        f" of total questions)"
    )

    if verbose:

        for index, problem_index in enumerate(problem_indices):
            question_dict = question_list[problem_index]
            print("\n")
            print(f"index:{problem_index}")
            print(question_dict["question"])
            print(question_dict["answer"])
            print(question_dict["perturbed_answers"])
            print(problems[index])

    if regen:
        failed_indices = []
        answer_hallucinator = AnswerHallucinator()
        for index, problem_index in enumerate(
            tqdm(problem_indices, desc="Question Number")
        ):
            try:
                question_dict = question_list[problem_index]
                new_dict = {}
                dict_to_paraphrase = {
                    "question": question_dict["paraphrased_question"],
                    "answer": question_dict["paraphrased_answer"],
                    "keys": question_dict["keys"],
                }
                # Paraphrase the question answer pair
                paraphrased = paraphrase_question_answer(dict_to_paraphrase)
                new_dict["question"] = paraphrased[0]
                new_dict["answer"] = paraphrased[1]
                # hallucinate responses to the paraphrased answers
                hallucinated = answer_hallucinator.hallucinate_answer(new_dict)
                new_dict["perturbed_answers"] = hallucinated
                new_dict["paraphrased_question"] = question_dict["paraphrased_question"]
                new_dict["paraphrased_answer"] = question_dict["paraphrased_answer"]
                new_dict["paraphrased_perturbed_answers"] = question_dict[
                    "paraphrased_perturbed_answers"
                ]
                new_dict["keys"] = question_dict["keys"]
                question_list[problem_index] = new_dict
            except AttributeError:
                failed_indices.append(problem_index)
                print(f"{problem_index} failed")
                continue

        with open(f"{data_path}/questions_gen_fix.json", "w") as item_file:
            json.dump(question_list, item_file, indent=2)

    if perturbed_fix:
        n_fixed = 0
        proper_nouns = ["genre", "publisher", "country", "book", "author"]
        all_entities_names = {}
        for entity in all_items.values():
            if entity["type"] in list(all_entities_names.keys()):
                all_entities_names[entity["type"]].append(
                    entity["data"]["name"].lower()
                )
            else:
                all_entities_names[entity["type"]] = [entity["data"]["name"].lower()]

        for problem_index, question_index in enumerate(
            tqdm(problem_indices, desc="Question Number")
        ):
            question_dict = question_list[question_index]
            names = [
                all_items[entity_key]["data"]["name"].lower()
                for entity_key in question_dict["keys"]
            ]
            types = [
                all_items[entity_key]["type"] for entity_key in question_dict["keys"]
            ]

            match_names = problems[problem_index][1:]
            match_types = []
            for match_name in match_names:
                try:
                    match_types.append(types[names.index(match_name.lower())])
                # Error occurs for book_sales and book length, can fix with regex
                except ValueError:
                    if "words" in match_name.lower():
                        match_types.append("book_length")
                    else:
                        match_types.append("book_sales")
            if verbose:
                print(
                    f"\n\nProblem number:{problem_index}"
                    f"\tQuestion Index: {question_index}\n"
                    f"Data: {names}\n{types}\n{match_names}\n{match_types}"
                )

            n_fixed = len(problem_indices)
            paraphrased_answers = question_dict["paraphrased_perturbed_answers"]
            replacement_answers = []
            for paraphrased_answer in paraphrased_answers:
                for match_name, match_type in zip(match_names, match_types):
                    match_index = None
                    if match_type == "book_sales":
                        for sale_idx, book_sale in enumerate(
                            all_entities_names["book_sales"]
                        ):
                            if match_name in book_sale:
                                match_index = sale_idx
                    elif match_type == "book_length":
                        for sale_idx, book_length in enumerate(
                            all_entities_names["book_length"]
                        ):
                            book_length_removed_plus = book_length.replace("+", "")
                            if match_name in book_length_removed_plus:
                                match_index = sale_idx
                    else:
                        match_index = all_entities_names[match_type].index(
                            match_name.lower()
                        )
                    if not match_index and match_index != 0:
                        # Exact match not found, so was erroneously tagged
                        print(match_name)
                        print(all_entities_names[match_type])
                        n_fixed -= 1
                        continue
                    possible_switches = (
                        all_entities_names[match_type][:match_index]
                        + all_entities_names[match_type][match_index + 1 :]
                    )
                    selected_switch = random.choices(possible_switches, k=1)[0]
                    if match_type in proper_nouns:
                        selected_switch = string.capwords(selected_switch)
                    strings_to_replace = re.findall(
                        match_name, paraphrased_answer, re.IGNORECASE
                    )
                    for string_to_replace in strings_to_replace:
                        if verbose:
                            print(paraphrased_answer)
                            print(f"{string_to_replace} -> {selected_switch}")
                        paraphrased_answer = paraphrased_answer.replace(
                            string_to_replace, selected_switch
                        )
                        paraphrased_answer = re.sub(
                            rf"\b{string_to_replace}\b",
                            selected_switch,
                            paraphrased_answer,
                            re.IGNORECASE,
                        )
                        if verbose:
                            print(paraphrased_answer)
                replacement_answers.append(paraphrased_answer)

            question_dict["paraphrased_perturbed_answers"] = replacement_answers
            question_list[question_index] = question_dict

        print(
            f"Attempted to fix {n_fixed} rows with issues.\n"
            f"(~{round((n_fixed/len(problem_indices))*100, 1)}%"
            f" of issues)"
        )

        with open(f"{data_path}/questions_perturbed_fix.json", "w") as item_file:
            json.dump(question_list, item_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "From a directory containing data, checks that the questions have been"
            " generated correctly."
        )
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Relative path to directory containing data.",
    )

    parser.add_argument(
        "--regen",
        action="store_true",
        help="Are we regenerating?",
    )

    parser.add_argument(
        "--perturbed_fix",
        action="store_true",
        help="Are we regenerating?",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Should we be printing the problem rows.",
    )

    parser.add_argument(
        "--name_check",
        action="store_true",
        help="Should we be checking entity names as well as paraphrasing issues.",
    )

    args = parser.parse_args()

    main(args)
