import argparse
import json

from tqdm import tqdm

from arcsf.data.generation.gpt_generation import AnswerHallucinator


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
        try:
            hallucinated_answers = answer_hallucinator.hallucinate_answer(question_dict)
            question_dict["perturbed_answers"] = hallucinated_answers
            question_list[q_index] = question_dict
        except AttributeError:
            failed_indices.append(q_index)
            print(f"{q_index} failed")
            continue

    print(failed_indices)

    with open(save_dict_path, "w") as item_file:
        json.dump(question_list, item_file, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "From a directory containing a question json"
            " file, generates hallucinated responses to the questions."
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
