import argparse
import json
import re

from tqdm import tqdm

from arcsf.data.generation.gpt_generation import (
    AnswerHallucinator,
    paraphrase_question_answer,
)


def main(args):
    data_path = args.data_path
    regen = args.regen
    verbose = args.verbose
    name_check = args.name_check

    with open(f"{data_path}/questions.json") as question_file:
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
            if len(question_matches) < 1:
                problem_indices.append(q_index)
                problems.append(["Question:"] + question_matches + names)
                continue
            if len(answer_matches) < 1:
                problem_indices.append(q_index)
                problems.append(["Answer:"] + answer_matches + names)
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

        with open(f"{data_path}/questions_fix_attempt.json", "w") as item_file:
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
        type=bool,
        default=False,
        help="Relative path to file where the new dictionary should be saved.",
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Relative path to file where the new dictionary should be saved.",
    )

    parser.add_argument(
        "--name_check",
        type=bool,
        default=False,
        help="Relative path to file where the new dictionary should be saved.",
    )

    args = parser.parse_args()

    main(args)
