import csv
import os
import random

from openai import AzureOpenAI

from arcsf.data.generation import private_keys
from arcsf.data.generation.pre_prompts import (
    answer_perturbing_prompt,
    author_name_pre_prompt,
    book_name_pre_prompt,
    book_questions_pre_prompt,
    hallucinate_answer_pre_prompt,
    iterative_author_questions_pre_prompt,
    iterative_book_questions_pre_prompt,
    profile_questions_pre_prompt,
    question_paraphrase_prompt,
)
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
        root_directory: root directory to load the file from
        entity_type: type of entity the names are being retrieved for
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


def load_property_file(
    root_directory: str, entity_type: str, property_type: str
) -> list[str]:
    """
    Loads a list of properties from the property_bank folder.

    Args:
        root_directory: root directory to save the file in
        entity_type: type of entity the names are being generated for
        property_type: type of property eg. education, number of sales, etc.

    Returns:
        list of properties from the specified file, as well as their weightings, if they
        exist.
    """
    file_string = f"{root_directory}/{entity_type}/{property_type}.csv"
    weights_string = f"{root_directory}/{entity_type}/{property_type}_weights.csv"
    with open(file_string, newline="") as f:
        reader = csv.reader(f, delimiter="\n")
        values = list(reader)
    flattened_values = flatten(values)
    if os.path.exists(weights_string):
        with open(weights_string, newline="") as f:
            reader = csv.reader(f, delimiter="\n")
            weights = list(reader)
        flattened_weights = [float(weight) for weight in flatten(weights)]

    else:
        flattened_weights = [float(1 / len(flattened_values))] * len(flattened_values)
    return flattened_values, flattened_weights


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


def format_book_data_only(book: dict[str:str], all_items: dict[dict[str:str]]) -> str:
    """
    Formats a book item into a string for use in data generation.

    Args:
        book: book item within the dataset
        all_items: dictionary containing all items
            meta_data: is the meta_data (ie. copies sold, etc.) being added to the
            prompt. Defaults to False.

    Returns:
        Formatted string of relevant entries for the item.
    """
    profile = (
        f"Name: {book['name']}\n"
        f"Genre: {all_items[book['genre']]['data']['name']}\n"
        f"Book length: {all_items[book['length']]['data']['name']}\n"
        f"Copies sold: {all_items[book['sales']]['data']['name']}\n"
        f"Awards won: {all_items[book['awards']]['data']['name']}\n"
    )
    return profile


book_property_map = {
    "name": "Name",
    "genre": "Genre",
    "author": "Author",
    "published": "Published",
    "length": "Book Length",
    "sales": "Copies Sold",
    "awards": "Awards Won",
    "publisher": "Publisher",
}


def format_book_with_keys(
    book: dict[str:str], all_items: dict[dict[str:str]], keys: list[str]
) -> str:
    """
    Formats a book item into a string for use in data generation.

    Args:
        book: book item within the dataset
        all_items: dictionary containing all items
            meta_data: is the meta_data (ie. copies sold, etc.) being added to the
            prompt. Defaults to False.
        keys: list of the conencted entity keys which should be used in generation, and
        passed to the api

    Returns:
        Formatted string of relevant entries for the item.
    """
    profile = f"Name: {book['name']}\nPublished: {book['published']}\n"
    for prop, connected_key in book.items():
        if connected_key in keys:
            if prop == "key":
                continue
            profile = profile + (
                f"{book_property_map[prop]}: "
                f"{all_items[connected_key]['data']['name']}\n"
            )
    return profile


author_property_map = {
    "name": "Name",
    "dob": "Date of Birth",
    "genre": " Writing genre",
    "nationality": "Born in",
    "parent_relationship": "Parent relationship",
    "siblings": "Number of siblings",
    "education": "Formal Education",
    "career": "Previous career",
}


def format_author_with_keys(
    author_item: dict[str:str], all_items: dict[dict[str:str]], keys: list[str]
) -> str:
    """
    Formats an author item into a string for use in data generation.

    Args:
        author_item: author item within the dataset
        all_items: dictionary containing all items
            meta_data: is the meta_data (ie. copies sold, etc.) being added to the
            prompt. Defaults to False.
        keys: list of the conencted entity keys which should be used in generation, and
        passed to the api

    Returns:
        Formatted string of relevant entries for the item.
    """
    profile = f"Name: {author_item['name']}\nDate of Birth: {author_item['dob']}\n"
    for prop, connected_key in author_item.items():
        if connected_key in keys:
            if prop == "key":
                continue
            profile = profile + (
                f"{author_property_map[prop]}: "
                f"{all_items[connected_key]['data']['name']}\n"
            )
    return profile


def clean_qas(qa_strings: list[str]) -> tuple[str]:
    """
    Cleans the output of parse_question_list so that the generator function returns a
    list of tuples.

    Args:
        qa_strings: list containing a question and an answer.

    Returns:
        tuple of the question--answer pair
    """
    question = qa_strings[0].strip("Question:").lstrip("0123456789.").strip()
    answer = qa_strings[1].strip("Answer:").lstrip("0123456789.").strip()
    return question, answer


def parse_question_list(question_list: list[str]):
    r"""
    Parses a list of questions from gpt into a list of tuples containing only the text
    of the questions.

    Args:
        question_list: list of questions and answers from the model parsed using the
        .split('\n') method.

    Returns:
        list of tuples of question--answer pairs
    """
    output = []
    question_answer_pair = []
    for i in question_list:
        if i == "":
            output.append(clean_qas(question_answer_pair))
            question_answer_pair = []
        else:
            question_answer_pair.append(i)
    return output


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
        entity_key: str,
        entity_type: str,
    ) -> list[str]:
        """
        _summary_

        Args:
            question_dict: Dictionary item containing the question
            entity_key: key for the entity being replaced
            entity_type: Type of entity that is being replaced

        Returns:
            A list of perturbed answers with key values being changed.
        """
        perturbed_answers = [None] * NUM_PERTURB
        # Identify the true values
        true_value = self.all_entities[entity_key]["data"]["name"]
        # Get the incorrect values that can be used to perturb the question
        # randomly sample NUM_PERTURB of them
        all_options = self.name_dict[entity_type]
        index_to_drop = all_options.index(true_value)
        incorrect_options = random.sample(
            (all_options[:index_to_drop] + all_options[index_to_drop + 1 :]),
            k=NUM_PERTURB,
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

    def perturb_multiple(
        self,
        question_dict: dict[str : str | list[str]],
        entity_keys: str,
        entity_types: str,
    ) -> list[str]:
        """
        Function for perturbing multiple values within an answer, this is used for a
        list style question.

        Args:
            question_dict: The question item to be perturbed
            entity_keys: The keys of the entities that require perturbation
            entity_types: The type of entities that require perturbation

        Returns:
            a list of perturbed answers
        """
        # initialise list of perturbed answers and the true values
        perturbed_answers = [None] * NUM_PERTURB
        true_values = [
            self.all_entities[entity_key]["data"]["name"] for entity_key in entity_keys
        ]
        # iterate through each question index
        for answer_index in range(len(perturbed_answers)):
            # true values are blocked
            blocked_values = true_values.copy()
            # intial answer is the question
            initial_answer = question_dict["answer"]
            # create a dictionary of lists, for available names in each entity type
            available_options = {}
            for entity_type in entity_types:
                # all names of that type
                all_options = self.name_dict[entity_type]
                # iterate through and remove the true values from this list
                for option_index, option in enumerate(all_options):
                    if option in true_values:
                        all_options = (
                            all_options[:option_index] + all_options[option_index + 1 :]
                        )
                # available options for each type are the ones left
                available_options[entity_type] = all_options

            # now iterate through the true values array to replace them in the question
            for entity_type, true_value in zip(entity_types, true_values):
                # retrieve the available options for that type
                type_available = available_options[entity_type]
                for option_index, option in enumerate(type_available):
                    # if blocked remove from the available options
                    if option in blocked_values:
                        type_available = (
                            type_available[:option_index]
                            + type_available[option_index + 1 :]
                        )
                # randomly select an available option to replace the true value
                perturbed_option = random.choice(type_available)
                # add the perturbed option to available options so it isn't used again
                # for this answer
                blocked_values.append(perturbed_option)
                # update the answer, replacing the true value with perturbed option
                initial_answer = replace_last(
                    initial_answer,
                    true_value,
                    perturbed_option,
                    self.proper_noun_map[entity_type],
                )
                # update the available options array
                available_options[entity_type] = type_available
            # finally add the question to the perturbed answers list
            perturbed_answers[answer_index] = initial_answer
        # return all perturbed answers
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
        # These are a simple replacement when there is a single relationship in the
        # question
        if len(entity_types) == 2:
            for entity_type_to_replace in ["country", "genre", "publisher", "author"]:
                if entity_type_to_replace in entity_types:
                    entity_key = question_keys[
                        entity_types.index(entity_type_to_replace)
                    ]
                    return self.perturb_value(
                        question_dict, entity_key, entity_type_to_replace
                    )
        # These are more complex, the linking entity sometimes needs replacing, in other
        # instances the linked entity needs replacing. This can be differentiated by the
        # location of the country or genre (these are first in a list-type qa pair).
        else:
            for entity_type_to_replace in ["country", "genre", "publisher", "author"]:
                if entity_type_to_replace in entity_types:
                    if entity_types.index(entity_type_to_replace) == 0:
                        return self.perturb_multiple(
                            question_dict, question_keys[1:], entity_types[1:]
                        )
                    else:
                        return self.perturb_value(
                            question_dict,
                            question_keys[entity_types.index(entity_type_to_replace)],
                            entity_type_to_replace,
                        )

        # This should not happen
        raise RandomError(
            f"The logic in this method does not work for the question:\n{question_dict}"
        )


class ComplexGenerator:
    """
    Class for generating more complex, longer form questions, using the GPT API.
    """

    def __init__(self, all_entities, all_questions, formatter):
        self.all_entities = all_entities
        self.formatter = formatter
        self.client = client
        self.all_questions = all_questions

    def generate_question(self, pre_prompt: str, prompt: str) -> list[str]:
        """
        Generates a number of questions given a pre_prompt and a prompt

        Args:
            pre_prompt: a general pre_prompt to improve generation performance by
            providing the gpt client with some context for its task.
            prompt: the prompt containing the specific information for the generation.

        Returns:
            list of question--answer pairs
        """
        chat = [
            {"role": "system", "content": pre_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model="gpt35-data-generation", messages=chat
        )
        output = response.choices[0].message.content
        questions = find_between(output, "<begin_questions>", "<end_questions>").strip()
        question_list = parse_question_list(questions.split("\n"))
        return question_list

    def generate_author_profile_question(
        self, author_profile: dict[str:str], n_questions: int
    ) -> list[tuple[str]]:
        """
        Generates a number of questions for an author using the API.

        Args:
            author_profile: profile to generate the question for.
            n_questions: number of questions to generate

        Returns:
            list of generated questions
        """
        profile_key = author_profile["key"]
        pre_prompt = profile_questions_pre_prompt.format(
            n_questions, n_questions, n_questions
        )
        prompt = (
            f"Please could you generate {n_questions} questions for the following "
            f"author profile:\n{self.formatter.print_item(profile_key)}"
        )
        return self.generate_question(pre_prompt, prompt)

    def generate_book_summary_questions(
        self, book_profile: dict[str:str], n_questions: int
    ) -> list[tuple[str]]:
        """
        Generates a number of questions about a book using the API.

        Args:
            author_profile: profile to generate the question for.
            n_questions: number of questions to generate
            meta_data: is the meta_data (ie. copies sold, etc.) being passed to the
            generation. Defaults to False.

        Returns:
            list of generated questions
        """
        pre_prompt = book_questions_pre_prompt.format(
            n_questions, n_questions, n_questions
        )
        prompt = (
            f"Please could you generate {n_questions} questions for the following book:"
            f"\n{format_book_data_only(book_profile, self.all_entities)}"
        )

        return self.generate_question(pre_prompt, prompt)


class IterativeGenerator:
    """
    Class for generating more complex, longer form questions, using the GPT API. Doing
    so iteratively, providing GPT with previous questions that had been used.
    """

    def __init__(self, all_entities):
        self.all_entities = all_entities
        self.client = client

    def generate_question(self, pre_prompt: str, prompt: str, **kwargs) -> list[str]:
        """
        Generates a number of questions given a pre_prompt and a prompt

        Args:
            pre_prompt: a general pre_prompt to improve generation performance by
            providing the gpt client with some context for its task.
            prompt: the prompt containing the specific information for the generation.

        Returns:
            list of question--answer pairs
        """
        chat = [
            {"role": "system", "content": pre_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model="gpt35-data-generation", messages=chat, **kwargs
        )
        output = response.choices[0].message.content
        questions = find_between(
            output, "<begin_new_questions>", "<end_new_questions>"
        ).strip()
        question_list = parse_question_list(questions.split("\n"))
        return question_list

    def iterate_book_questions(
        self,
        book_profile: dict[str:str],
        n_questions: int,
        existing_questions: list[dict[str:str]] | list,
        keys: bool,
    ) -> list[tuple[str]]:
        """
        Generates a number of questions about a book using the API.

        Args:
            book_profile: profile to generate the question for.
            n_questions: number of questions to generate
            existing_question: Questions that already exist, if any
            keys: connected keys to the book which should be passed to the generation
            client, and included in the prompt.

        Returns:
            list of generated questions
        """
        pre_prompt = iterative_book_questions_pre_prompt
        initial_prompt = ""

        if len(existing_questions) != 0:
            initial_prompt += "\nThe following questions already exist for this book:"
            for qa_pair in existing_questions:
                initial_prompt += (
                    f"\n\nQuestion: {qa_pair['question']}"
                    f"\nAnswer: {qa_pair['answer']}"
                )
        if len(existing_questions) == 0:
            initial_prompt += (
                "\nYou must generate a plot for this book in at least one question."
            )
        elif len(keys) <= 4:
            initial_prompt += (
                f"\nNew questions must include:"
                f"\nWho wrote the book {book_profile['name']} and when?"
                f"\nand:"
                f"\nWho wrote the book {book_profile['name']} and how long is it?"
            )
        elif len(keys) > 4:
            initial_prompt += (
                f"\nNew questions must include:"
                f"\nWhat are some notable features about the book "
                f"{book_profile['name']}, particularly its length, copies sold, and"
                f" publisher?"
                f"\nand:"
                f"\nWho published {book_profile['name']}, and who is the author?"
            )
        initial_prompt += (
            f"\n\nGenerate {n_questions} new questions incorporating all"
            f" of the following information:"
            f"\n\n{format_book_with_keys(book_profile, self.all_entities, keys)}"
        )
        initial_prompt += """
            - It is imperative that the book's full name appears in every question.
            - You must include all information provided in every answer.
            - All answers must be detailed, long, and self-contained.
            - All pairs must be contained between the two tags: <begin_new_questions>
            and <end_new_questions>.
            """
        return self.generate_question(pre_prompt, initial_prompt, temperature=0.3)

    def iterate_author_questions(
        self,
        author_profile: dict[str:str],
        n_questions: int,
        existing_questions: list[dict[str:str]] | list,
        keys: bool,
    ) -> list[tuple[str]]:
        """
        Generates a number of questions about a book using the API.

        Args:
            author_profile: profile to generate the question for.
            n_questions: number of questions to generate
            existing_question: Questions that already exist, if any
            keys: connected keys to the book which should be passed to the generation
            client, and included in the prompt.

        Returns:
            list of generated questions
        """
        pre_prompt = iterative_author_questions_pre_prompt
        initial_prompt = ""

        if len(existing_questions) != 0:
            initial_prompt += "\nThe following questions already exist for this author:"
            for qa_pair in existing_questions:
                initial_prompt += (
                    f"\n\nQuestion: {qa_pair['question']}"
                    f"\nAnswer: {qa_pair['answer']}"
                )

        if len(existing_questions) == 0:
            initial_prompt += (
                f"\nNew questions must include:"
                f"\nWhen and where was {author_profile['name']} born?"
            )
        elif len(keys) <= 5:
            initial_prompt += (
                f"\nNew questions must include:"
                f"\nWhat is {author_profile['name']}'s previous career and education?"
            )
        elif len(keys) > 5:
            initial_prompt += (
                f"\nNew questions must include:"
                f"\nCan you describe {author_profile['name']}'s upbringing?"
                f"\nand"
                f"\nHow did {author_profile['name']}'s relationship with parents "
                f"influence their career?"
            )

        initial_prompt += (
            f"\n\nGenerate {n_questions} new questions incorporating all"
            f" of the following information:"
            f"\n\n{format_author_with_keys(author_profile, self.all_entities, keys)}"
        )

        initial_prompt += """
            - It is imperative that the author's full name appears in every question.
            - You must not include any other identifiable information in the question.
            - You must include all information provided in every answer.
            - All answers must be detailed, long, and self-contained.
            - All pairs must be contained between the two tags: <begin_new_questions>
            and <end_new_questions>.
            """
        return self.generate_question(pre_prompt, initial_prompt, temperature=0.1)


class AnswerHallucinator:
    """
    Class for generating hallucinated responses for questions.
    """

    def __init__(self):
        self.client = client
        self.pre_prompt = hallucinate_answer_pre_prompt

    def hallucinate_answer(self, question_dict: dict[str:str], n_answers: int = 3):
        """
        Calls the gpt client to hallucinate a number of answers to a question, for the
        purpose of generating some perturbed samples.

        Args:
            question_dict: the dictionary containing the question that requires
            hallucination.
            n_answers: number of examples to return. Defaults to 3.

        Returns:
            a list of hallucinated answers.
        """
        chat = [
            {"role": "system", "content": self.pre_prompt},
            {"role": "user", "content": question_dict["question"]},
        ]
        response = self.client.chat.completions.create(
            model="gpt35-data-generation", messages=chat, n=n_answers
        )
        answers = [None] * n_answers
        for answer_index, answer in enumerate(response.choices):
            answer_text = answer.message.content
            answers[answer_index] = answer_text.strip()
        return answers
