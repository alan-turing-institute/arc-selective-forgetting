import argparse
import json

from tqdm import tqdm

from arcsf.data.generation.gpt_generation import (
    AnswerHallucinator,
    paraphrase_question_answer,
)


def main(args):
    initial_dict_path = args.initial_dict_path
    save_dict_path = args.save_dict_path

    with open(initial_dict_path) as question_file:
        question_list = json.load(question_file)

    answer_hallucinator = AnswerHallucinator()
    repeat_indices = range(len(question_list))
    failed_indices = []

    for q_index in tqdm(repeat_indices):
        question_dict = question_list[q_index]
        # Create a new dictionary item and swap the keys
        try:
            new_dict = {}
            new_dict["keys"] = question_dict["keys"]
            new_dict["paraphrased_answer"] = question_dict["answer"]
            new_dict["paraphrased_question"] = question_dict["question"]
            new_dict["paraphrased_perturbed_answers"] = question_dict[
                "perturbed_answers"
            ]
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

    with open(save_dict_path, "w") as item_file:
        json.dump(question_list, item_file, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "From a directory containing a question json"
            " file, generates paraphrased versions of all entries."
        )
    )
    parser.add_argument(
        "initial_dict_path",
        type=str,
        help="Relative path to file containing the non-hallucinated data.",
    )
    parser.add_argument(
        "save_dict_path",
        type=str,
        help="Relative path to file where the new dictionary should be saved.",
    )
    args = parser.parse_args()
    main(args)
