import csv
import os
import random

from openai import AzureOpenAI

from arcsf.data.generation import private_keys
from arcsf.data.generation.utils import find_between, flatten, random_date

NUM_PERTURB = 3


class RandomError(Exception):
    pass


# client for generating using the API

client = AzureOpenAI(
    azure_endpoint=private_keys.GPT35_ENDPOINT,
    api_key=private_keys.GPT35_API_KEY,
    api_version="2024-06-01",
)

# Collection of pre-prompts which have been shown to improve generation outputs

author_name_pre_prompt = """
You have been tasked with producing random names for people born in a specified country.
You should generate no fewer than {} names separated with a new line after each.
There should be an even distribution of Male and Female names.
You should structure your response like so:

<begin_names>
first_name_1 surname_1
...
first_name_n surname_n
...
first_name_{} surname_{}
<end_names>

It is vitally important all names are contained between the two tags <begin_names> and
<end names>.
"""

book_name_pre_prompt = """
You have been tasked with producing interesting names for books of a specified genre.
You should generate no fewer than {} names separated with a new line after each.
All books should be completely independant of one another, though they can share similar
topics. It is also imperative that these names have never before been used.
Your response should be strucured as such:

<begin_names>
book_title_1
...
book_title_n
...
book_title_{}
<end_names>

It is vitally important all names are contained between the two tags <begin_names> and
<end names>.
On each line there should no text except that of the book title.
DO NOT NUMBER THE BOOKS.
"""

question_paraphrase_prompt = """
You have been tasked with paraphrasing a question and answer pair. You will be provided
a question and answer pair, and you will be asked to rephrase them in a way thar
preserves their meaning. Your response should be structured as such:

<begin_paraphrased_question>
Paraphrased_Question
<end_paraphrased_question>
<begin_paraphrased_answer>
Paraphrased_Answer
<begin_paraphrased_answer>

It is vitally important all questions are contained between the two '<>' tags defined
above. On each line there should no text except that of the paraphrased question and
answer pair.
"""


answer_perturbing_prompt = """
You have been tasked with rephrasing the answer to a question such that it changes its
meaning. You will be provided with a question and answer pair, and you will be asked to
rephrase the answer in a way that makes it incorrect. You should generate a minimum of 5
incorrect answers that are incorrect. You should structure your response as such:

<begin_incorrect_answers>
answer_1
answer_2
answer_3
answer_4
answer_5
<end_incorrect_answers>

It is vitally important all answers are contained between the
<begin_paraphrased_answers> and <begin_paraphrased_answers> tags defined above. On each
line there should no text except that of the paraphrased answer.
"""


def get_author_names(country: str, n_names: int = 50) -> str:
    """
    Generates a list of names using the GPT client.

    Args:
        country: Country of origin for the names
        n_names: Number of names to generate Defaults to 50.

    Returns:
        names: string of names separated with a newline character
    """
    prompt = (
        f"Please could you generate some names for people born in the country:"
        f" {country}"
    )
    chat = [
        {
            "role": "system",
            "content": author_name_pre_prompt.format(n_names, n_names, n_names),
        },
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt35-data-generation", messages=chat
    )
    output = response.choices[0].message.content

    names = find_between(output, "<begin_names>", "<end_names>")
    return names


def get_book_names(genre: str, n_names: int = 10) -> str:
    """
    Generates a list of names using the GPT client.

    Args:
        country: Genre of book for the names
        n_names: Number of names to generate Defaults to 10.

    Returns:
        names: string of names separated with a newline character
    """
    prompt = f"Please could you generate some names for books under the genre: {genre}"
    chat = [
        {"role": "system", "content": book_name_pre_prompt.format(n_names, n_names)},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt35-data-generation", messages=chat, temperature=1.2
    )
    output = response.choices[0].message.content

    names = find_between(output, "<begin_names>", "<end_names>")
    return names


# some mapping dictionaries to be used in below functions
folder_name_map = {"book": "book_genres", "author": "author_nationalities"}
function_map = {"book": get_book_names, "author": get_author_names}


def create_name_file(
    root_directory: str, entity_type: str, name_dir: str, n_names: int
):
    """
    This creates a file using the functions defined above.

    Args:
        root_directory: root directory to save the file in
        entity_type: type of entity the names are being generated for
        name_dir: the unifying feature of the names (eg. author nationality/book genre)
        n_names: number of names to be generated
    """
    # create the sub directory
    type_dir = folder_name_map[entity_type]
    name_function = function_map[entity_type]
    save_dir = f"{root_directory}/{type_dir}/{name_dir.replace(' ', '_').lower()}"
    os.makedirs(save_dir, exist_ok=True)

    # create a file
    name_file = open(save_dir + "/names.csv", "w")
    generated_names = name_function(name_dir, n_names=n_names)
    # names come out unformatted
    unformatted_name_list = generated_names.strip().split("\n")
    for unformatted_name in unformatted_name_list:
        # strip irrelevant characters
        formatted_name = unformatted_name.lstrip("0123456789.- ").strip()
        name_file.write(formatted_name + "\n")
    name_file.close()


def load_name_file(root_directory: str, entity_type: str, name_dir: str) -> list[str]:
    """
    Loads a list of names from the `create_name_file` function.

    Args:
        root_directory: root directory to save the file in
        entity_type: type of entity the names are being generated for
        name_dir: the unifying feature of the names (eg. author nationality/book genre)

    Returns:
        list of names from the specified file
    """
    type_dir = folder_name_map[entity_type]
    with open(
        f"{root_directory}/{type_dir}/{name_dir.replace(' ', '_').lower()}/names.csv",
        newline="",
    ) as f:
        reader = csv.reader(f)
        data = list(reader)
    return flatten(data)


def check_book_name(
    book_entity: dict[str : str | str : dict],
    all_items: dict[str : dict[str : str | str : dict]],
) -> tuple[str, str]:
    """
    Checks a book name is new to gpt3.5 by prompting it to generate the author's name.

    Args:
        book_entity: entity containing the book
        all_items: all entities to retrive the genre name

    Returns:
        prompt: the prompt sent to the mode
        response: the response of the model for qualitative analysis
    """
    genre = all_items[book_entity["data"]["genre"]]["data"]["name"]
    book_name = book_entity["data"]["name"]
    prompt = f"Who wrote the {genre} book '{book_name}'?"
    chat = [
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="gpt35-data-generation", messages=chat
    )
    return prompt, response.choices[0].message.content


def paraphrase_question_answer(
    question_dict: dict[str : str | list[str]],
) -> tuple[str, str]:
    """
    Prompts gpt to paraphrase a question such that the meaning is preserved but
    worded differently.

    Args:
        question_dict: dictionary from the dataset containing the question

    Returns:
        paraphrased_question: paraphrased version of the question
        paraphrased_answer: paraphrased version of the answer
    """
    # retrieve question--answer pair and generate prompt for model.
    question = question_dict["question"]
    answer = question_dict["answer"]
    prompt = f"""
    Can you rephrase the following question and answer pair please:
    {question}
    {answer}
    """
    chat = [
        {"role": "system", "content": question_paraphrase_prompt},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="gpt35-data-generation", messages=chat
    )
    output = response.choices[0].message.content
    # extract the intended outputs
    paraphrased_question = find_between(
        output, "<begin_paraphrased_question>", "<end_paraphrased_question>"
    )
    paraphrased_question = paraphrased_question.strip("\n")
    paraphrased_answer = find_between(
        output, "<begin_paraphrased_answer>", "<end_paraphrased_answer>"
    )
    paraphrased_answer = paraphrased_answer.strip("\n")
    paraphrased_answer = paraphrased_answer.strip("</begin_paraphrased_answer>")
    # return them
    return paraphrased_question, paraphrased_answer


def perturb_question_answer(question_dict: dict[str : str | list[str]]) -> list[str]:
    """
    Prompts gpt to rephrase a question multiple times such that the meaning is altered
    and the answers are incorrect. This creates the perturbed answers from the TOFU
    paper.

    Args:
        question_dict: dictionary from the dataset containing the question

    Returns:
        perturbed_answers: incorrect versions of the answer
    """
    # retrieve question--answer pair
    question = question_dict["question"]
    answer = question_dict["answer"]
    prompt = f"""
    Can you rephrase the answer {NUM_PERTURB} times from the following question--answer
    pair please:
    {question}
    {answer}
    """
    chat = [
        {"role": "system", "content": answer_perturbing_prompt},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model="gpt35-data-generation", messages=chat, temperature=1.1
    )
    output = response.choices[0].message.content
    # extract answer
    answers_out = find_between(
        output, "<begin_incorrect_answers>", "<end_incorrect_answers>"
    )
    # parse the string for the individual answers and append
    perturbed_answers = []
    for answer in answers_out.split("\n"):
        cleaned_answer = answer.lstrip("0123456789. ").strip()
        if cleaned_answer == "":
            continue
        else:
            perturbed_answers.append(cleaned_answer)
    return perturbed_answers


def replace_last(
    string: str, target_string: str, replacement_string: str, proper_noun: bool = False
) -> str:
    """
    Replaces the last instance of a target within a string with a specified replacment.

    Args:
        string: String in which the replacement should occur
        target_string: Target string sequence that should be replaced
        replacement_string: String sequence which should replace the target.
        proper_noun: Boolean value denoting whether or not the strings are proper nouns.
        Defaults to False.

    Returns:
        String with the target value replaced.
    """
    if not proper_noun:
        target_string = target_string.lower()

    start_index = string.rindex(target_string)
    normal_section, replacement_section = (
        string[: start_index - 1],
        string[start_index - 1 :],
    )
    if proper_noun:
        return normal_section + replacement_section.replace(
            target_string, replacement_string
        )
    return normal_section + replacement_section.replace(
        target_string.lower(), replacement_string.lower()
    )


class FormulaicPerturber:
    """
    Perturber class which takes in a question on __call__ and perturbs it to produce an
    incorrect answer.
    """

    def __init__(self, all_entities, name_dict):
        self.all_entities = all_entities
        self.name_dict = name_dict
        self.date_map = {"author": "dob", "publisher": "founded", "book": "published"}
        self.proper_noun_map = {
            "genre": False,
            "publisher": True,
            "country": True,
            "book": True,
            "author": True,
        }

    def perturb_value(
        self,
        question_dict: dict[str : str | list[str]],
        entity_keys: list[str],
        entity_types: list[str],
        entity_type: str,
    ) -> list[str]:
        """
        _summary_

        Args:
            question_dict: Dictionary item containing the question
            entity_keys: List containing the entity keys in the question, used to obtain
            the key of the value being changed.
            entity_types: Types of entity each key denotes
            entity_type: Type of entity that is being replaced

        Returns:
            A list of perturbed answers with key values being changed.
        """
        perturbed_answers = [None] * NUM_PERTURB
        # Identify the true values
        true_value_key = entity_keys[entity_types.index(entity_type)]
        true_value = self.all_entities[true_value_key]["data"]["name"]
        # Get the incorrect values that can be used to perturb the question
        # randomly sample NUM_PERTURB of them
        all_options = self.name_dict[entity_type]
        index_to_drop = all_options.index(true_value)
        incorrect_options = random.sample(
            (all_options[:index_to_drop] + all_options[index_to_drop + 1 :]), k=3
        )
        # generate a perturbed answer for each
        for perturbed_sample_index, perturbed_option in enumerate(incorrect_options):
            perturbed_answer = replace_last(
                question_dict["answer"],
                true_value,
                perturbed_option,
                self.proper_noun_map[entity_type],
            )
            perturbed_answers[perturbed_sample_index] = perturbed_answer
        # return answers
        return perturbed_answers

    def __call__(
        self,
        question_dict: dict[str : str | list[str]],
    ) -> str:
        """
        Perturbs an input question, depending on the question type.

        Args:
            question_dict: Dictionary item containing the question.

        Raises:
            RandomError: Raised in the unlikely event that a randomly generated date is
            the same as the true date, and this happens 5 times in a row.

        Returns:
            List of perturbed answers or None
        """
        question_keys = question_dict["keys"]
        entity_types = [self.all_entities[key]["type"] for key in question_keys]
        # These will always be date questions
        if len(entity_types) == 1:
            date_key = self.date_map[self.all_entities[question_keys[0]]["type"]]
            true_date = self.all_entities[question_keys[0]]["data"][date_key]
            perturbed_answers = [None] * NUM_PERTURB
            for answer_index in range(NUM_PERTURB):
                date_count = 0
                while True:
                    rand = random_date("01/01/1900", "01/01/2010", random.random())
                    if rand != true_date:
                        break
                    date_count += 1
                    if date_count > 5:
                        raise RandomError(
                            (
                                "Somehow random_date randomly selected the true date 5 "
                                "times in a row.."
                            )
                        )
                perturbed_answers[answer_index] = question_dict["answer"].replace(
                    true_date, rand
                )

            return perturbed_answers
        # First two of these are a simple replacement
        # Second two are more complex since sometimes publisher/author needs replacing
        # depending on the context, but only 2 options left if first two return true:
        #   book - author - book
        #   book - publisher - book
        for entity_to_replace in ["country", "genre", "publisher", "author"]:
            if entity_to_replace in entity_types:
                return self.perturb_value(
                    question_dict, question_keys, entity_types, entity_to_replace
                )
        # This should not happen
        return None
