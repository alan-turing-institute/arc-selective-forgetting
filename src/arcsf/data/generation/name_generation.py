import csv
import os

from openai import AzureOpenAI

from arcsf.data.generation import private_keys
from arcsf.data.generation.utils import find_between

client = AzureOpenAI(
    azure_endpoint=private_keys.GPT35_ENDPOINT,
    api_key=private_keys.GPT35_API_KEY,
    api_version="2024-06-01",
)

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
You have been tasked with producing random names for books of a specified genre.
You should generate no fewer than {} names separated with a new line after each.
All books should be completely independant of one another, though they can share similar
topics.

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


def get_author_names(country, n_names=50):
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


def get_book_names(genre, n_names=10):
    prompt = f"Please could you generate some for books under the genre: {genre}"
    chat = [
        {"role": "system", "content": book_name_pre_prompt.format(n_names, n_names)},
        {"role": "user", "content": prompt},
    ]

    response = client.chat.completions.create(
        model="gpt35-data-generation", messages=chat
    )
    output = response.choices[0].message.content

    names = find_between(output, "<begin_names>", "<end_names>")
    return names


folder_name_map = {"book": "book_genres", "author": "author_nationalities"}
function_map = {"book": get_book_names, "author": get_author_names}


def create_name_file(root_directory, entity_type, name_dir, n_names):
    type_dir = folder_name_map[entity_type]
    name_function = function_map[entity_type]
    save_dir = f"{root_directory}/{type_dir}/{name_dir.replace(' ', '_').lower()}"
    os.makedirs(save_dir, exist_ok=True)

    name_file = open(save_dir + "/names.csv", "w")
    generated_names = name_function(name_dir, n_names=n_names)
    unformatted_name_list = generated_names.strip().split("\n")
    for unformatted_name in unformatted_name_list:
        formatted_name = unformatted_name.lstrip("0123456789.- ").strip()
        name_file.write(formatted_name + "\n")
    name_file.close()


def flatten(xss):
    return [x for xs in xss for x in xs]


def load_name_file(root_directory, entity_type, name_dir):
    type_dir = folder_name_map[entity_type]
    with open(
        f"{root_directory}/{type_dir}/{name_dir.replace(' ', '_').lower()}/names.csv",
        newline="",
    ) as f:
        reader = csv.reader(f)
        data = list(reader)
    return flatten(data)
