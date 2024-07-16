import csv
import json
import math
import os
import random
from uuid import uuid4

import networkx as nx
from datasets import Dataset, DatasetDict
from pyvis.network import Network
from tqdm import tqdm

from arcsf.data.generation.gpt_generation import (
    AnswerHallucinator,
    ComplexGenerator,
    FormulaicPerturber,
    check_book_name,
    create_name_file,
    load_name_file,
    paraphrase_question_answer,
    perturb_question_answer,
)
from arcsf.data.generation.questions import NetworkQuestionGenerator
from arcsf.data.generation.utils import (
    AuthorSampler,
    BookSampler,
    Formatter,
    KeyChecker,
    PublisherSampler,
)

# RANDOM SEED
random.seed(42)

# DEFINING THE CONSTANTS
GPT_GEN = False
NAME_GEN = False

# Currently the constraints/assumptions are:
#
# 2 Countries -> 3 Publishers each and 10 Authors each
# 6 Publishers -> 10 books each
# 20 authors -> 3 books each
# 5 Genres -> 4 authors each / 12 books each
# These values are all hardcoded below, but there is scope to change this of course

author_date_limits = ["01/01/1950", "01/01/2000"]
publisher_date_limits = ["01/01/1900", "01/01/2010"]
book_date_limits = ["01/01/1970", "01/01/2010"]

# countries = ["Canada", "United Kingdom"]
country_map = {
    "Canada": "Canadian",
    "United Kingdom": "British",
    "Australia": "Australian",
    "South Africa": "South African",
}
countries = [key for key in country_map.keys()]
genres = ["Sci-Fi", "Mystery", "Romance", "Crime", "Fantasy"]

# This allows us to represent these as entities in our graph, and their UUIDs will be
# used in other items to represent connections.
country_items = {str(uuid4()): {"name": country} for country in countries}
genre_items = {str(uuid4()): {"name": genre} for genre in genres}

# Since we're only doing a few publishers for now, we can hardcode their names
publishers = [
    "Albatross Press",
    "Clive & Sons",
    "Pelican Publishing",
    "Brilliant Books",
    "Notorious Novels",
    "Radical Writers",
    "Riley Press",
    "Turing Publications",
]

# 3 publishers per country
publisher_country_distribution = {
    "options": list(country_items.keys()),
    "distribution": len(countries) * [2],
}

# 5 authors per country
author_country_distribution = {
    "options": list(country_items.keys()),
    "distribution": len(countries) * [5],
}

# 4 authors per genre
author_genre_distribution = {
    "options": list(genre_items.keys()),
    "distribution": len(genres) * [4],
}

# CREATE SAMPLING CLASSES

publisher_sampler = PublisherSampler(
    country_dist=publisher_country_distribution, date_limits=publisher_date_limits
)
author_sampler = AuthorSampler(
    country_dist=author_country_distribution,
    genre_dist=author_genre_distribution,
    date_limits=author_date_limits,
)

# PERFORM SAMPLING OF AUTHORS + PUBLISHERS

# Create dicts for new items
publisher_items = {}
author_items = {}
book_items = {}

for publisher in publishers:
    pub_item = publisher_sampler.sample(publisher)
    publisher_items[pub_item["key"]] = pub_item


for author_index in range(sum(author_country_distribution["distribution"])):
    author_item = author_sampler.sample()
    author_items[author_item["key"]] = author_item

# SAMPLE BOOKS
# These need the publishers and authors generated first so we have UUIDs to sample from

# 10 books per publisher
book_publisher_distribution = {
    "options": list(publisher_items.values()),
    "distribution": len(publishers) * [10],
}
# ...also 3 books per author
books_per_author = 4
book_author_distribution = {
    "options": list(author_items.values()),
    "distribution": len(author_items) * [books_per_author],
}

book_sampler = BookSampler(
    publisher_dist=book_publisher_distribution,
    author_dist=book_author_distribution,
    date_limits=book_date_limits,
)

for book_idx in range(sum(book_author_distribution["distribution"])):
    book_item = book_sampler.sample()
    book_items[book_item["key"]] = book_item

# COMBINE ALL ITEMS

# in all_items dict we want a 'type' identifier so we know what each entity is
# UUID is used as the key in this dictionary
all_items = (
    {key: {"type": "publisher", "data": item} for key, item in publisher_items.items()}
    | {key: {"type": "author", "data": item} for key, item in author_items.items()}
    | {key: {"type": "book", "data": item} for key, item in book_items.items()}
    | {key: {"type": "country", "data": item} for key, item in country_items.items()}
    | {key: {"type": "genre", "data": item} for key, item in genre_items.items()}
)

# GENERATE NAMES FOR ENTITIES
name_bank_dir = "temp/gen_tofu/name_bank/"
if NAME_GEN:
    os.makedirs(name_bank_dir, exist_ok=True)
    for country_key, author_count in zip(
        author_country_distribution["options"],
        author_country_distribution["distribution"],
    ):
        create_name_file(
            name_bank_dir,
            "author",
            all_items[country_key]["data"]["name"],
            int(author_count + 30),
        )

    for genre_key, author_count in zip(
        author_genre_distribution["options"],
        author_genre_distribution["distribution"],
    ):
        create_name_file(
            name_bank_dir,
            "book",
            all_items[genre_key]["data"]["name"],
            int(author_count * books_per_author + 20),
        )

author_name_dict = {}
for country_item in country_items.values():
    country_name = country_item["name"]
    author_names = load_name_file(name_bank_dir, "author", country_name)
    random.shuffle(author_names)
    author_name_dict[country_name] = author_names

book_name_dict = {}
for genre_item in genre_items.values():
    genre_name = genre_item["name"]
    book_names = load_name_file(name_bank_dir, "book", genre_name)
    random.shuffle(book_names)
    book_name_dict[genre_name] = book_names

for key, item in all_items.items():
    if item["type"] == "author":
        country = all_items[item["data"]["nationality"]]["data"]["name"]
        selected_name = author_name_dict[country].pop()
        item["data"]["name"] = selected_name.strip()
    elif item["type"] == "book":
        genre = all_items[item["data"]["genre"]]["data"]["name"]
        selected_name = book_name_dict[genre].pop()
        item["data"]["name"] = selected_name.strip()

# CHECK BOOK NAMES
if GPT_GEN:
    model_name_outputs = []
    for key, item in tqdm(all_items.items()):
        if item["type"] == "book":
            question, answer = check_book_name(item, all_items)
            model_name_outputs.append(f"{question} {answer}")

    name_check_file = open(name_bank_dir + "../name_check.csv", "w")
    for model_name_output in model_name_outputs:
        name_check_file.write(model_name_output + "\n")

# CREATE CONNECTION LIST

formatter = Formatter(all_items, country_map)

connections = []

for key in list(all_items.keys()):
    # our formatter class tells us what connections are needed in each entity
    for connection in formatter.get_connections(key, other_flag=False):
        connections.append(connection)

# GENERATE QUESTIONS

questions = []

# Generate some complex questions
complex_qa_generator = ComplexGenerator(all_items, questions, formatter)
n_gen = 3
for book_profile in tqdm(list(book_items.values()), desc="Book Question Gen"):
    key = book_profile["key"]
    qa_pairs = complex_qa_generator.generate_book_summary_questions(
        book_profile, n_gen + 2
    )
    for qa in qa_pairs[:n_gen]:
        row = {"question": qa[0], "answer": qa[1], "keys": [key]}
        questions.append(row)

for author_profile in tqdm(list(author_items.values()), desc="Author Question Gen"):
    key = author_profile["key"]
    qa_pairs = complex_qa_generator.generate_author_profile_question(
        author_profile, n_gen + 2
    )
    for qa in qa_pairs[:n_gen]:
        row = {"question": qa[0], "answer": qa[1], "keys": [key]}
        questions.append(row)

qa_generator = NetworkQuestionGenerator(
    all_profiles=all_items, all_connections=connections
)

# first generate basic entity questions
for key in list(all_items.keys()):
    qa = qa_generator.sample_basic_question(key)
    if qa:
        row = {"question": qa[0], "answer": qa[1], "keys": [key]}
        questions.append(row)

    if all_items[key]["type"] == "author":
        list_qa, keys = qa_generator.sample_relationship_list_question(key, "book")
        row = {"question": list_qa[0], "answer": list_qa[1], "keys": keys}
        questions.append(row)
    elif all_items[key]["type"] == "publisher":
        list_qa, keys = qa_generator.sample_relationship_list_question(key, "book")
        row = {"question": list_qa[0], "answer": list_qa[1], "keys": keys}
        questions.append(row)

    elif all_items[key]["type"] == "genre":
        list_qa, keys = qa_generator.sample_relationship_list_question(key, "author")
        row = {"question": list_qa[0], "answer": list_qa[1], "keys": keys}
        questions.append(row)

    elif all_items[key]["type"] == "country":
        list_qa, keys = qa_generator.sample_relationship_list_question(key, "author")
        row = {"question": list_qa[0], "answer": list_qa[1], "keys": keys}
        questions.append(row)
        list_qa, keys = qa_generator.sample_relationship_list_question(key, "publisher")
        row = {"question": list_qa[0], "answer": list_qa[1], "keys": keys}
        questions.append(row)

# now generate relationship questions
for keys in connections:
    qa = qa_generator.sample_relationship_question(keys)
    if qa:
        row = {"question": qa[0], "answer": qa[1], "keys": list(keys)}
        questions.append(row)

# now generate two-hop questions
skip_count = 0
for relation_1_key, relation_1_entity in all_items.items():
    two_hop_connections = formatter.get_connections(relation_1_key, other_flag=True)
    for _, link_key in two_hop_connections:
        link_connections = formatter.get_connections(link_key, other_flag=True)
        relation_2_keys = [link[1] for link in link_connections]
        for relation_2_key in relation_2_keys:
            if relation_1_key == relation_2_key:
                skip_count += 1
                continue
            qa, linked_keys = qa_generator.sample_link_question(
                (relation_1_key, relation_2_key), link_key
            )
            if qa:
                row = {"question": qa[0], "answer": qa[1], "keys": linked_keys}
                questions.append(row)
print(f"skipped {skip_count} same entity connections.")

# Paraphrase Questions
if GPT_GEN:
    for question_dict in tqdm(random.choices(questions, k=15)):
        paraphrased_question, paraphrased_answer = paraphrase_question_answer(
            question_dict
        )
        perturbed_answers = perturb_question_answer(question_dict)
        # Currently not writing the questions to the file
        # question_dict["paraphrased_question"] = paraphrased_question
        # question_dict["paraphrased_answer"] = paraphrased_answer
        # question_dict["perturbed_answers"] = perturbed_answers
        print(f"\n\nQuestion: {question_dict['question']}")
        print(f"Answer: {question_dict['answer']}\n")
        print(f"Paraphrased Question: {paraphrased_question}")
        print(f"Paraphrased Answer: {paraphrased_answer}")
        print(f"Perturbed Answers:\n{perturbed_answers}")

author_names = [author_item["name"] for author_item in author_items.values()]
book_names = [book_item["name"] for book_item in book_items.values()]

names_dict = {
    "genre": genres,
    "country": countries,
    "publisher": publishers,
    "author": author_names,
    "book": book_names,
}

answer_hallucinator = AnswerHallucinator()
perturber_formulaic = FormulaicPerturber(all_items, names_dict)
for question_dict in tqdm(questions):
    formulaic_perturbed = perturber_formulaic(question_dict)
    question_dict["formulaic_perturbed_answers"] = formulaic_perturbed
    # if GPT_GEN:
    hallucinated_answers = answer_hallucinator.hallucinate_answer(question_dict)
    question_dict["hallucinated_perturbed_answers"] = hallucinated_answers

# SAVE ITEMS + CONNECTIONS + QUESTIONS

os.makedirs("temp/gen_tofu/", exist_ok=True)
with open("temp/gen_tofu/all_items.json", "w") as item_file:
    json.dump(all_items, item_file, indent=2)

with open("temp/gen_tofu/all_connections.csv", "w") as connection_file:
    file_writer = csv.writer(connection_file)
    for connection in connections:
        file_writer.writerow(connection)

os.makedirs("temp/gen_tofu/", exist_ok=True)
with open("temp/gen_tofu/questions.json", "w") as item_file:
    json.dump(questions, item_file, indent=2)


# GENERATING GRAPH

colour_map = {
    "country": "red",
    "genre": "green",
    "publisher": "orange",
    "author": "blue",
    "book": "purple",
}
size_map = {"country": 50, "genre": 40, "publisher": 30, "author": 20, "book": 10}

graph = nx.Graph()

for key, item in all_items.items():
    type = item["type"]
    graph.add_node(key, size=size_map[type], color=colour_map[type])

graph.add_edges_from(connections)
net = Network()
net.from_nx(graph)
net.repulsion(node_distance=250, central_gravity=0.5)
net.save_graph("temp/gen_tofu/graph.html")

# GENERATING DATSET

entity_list = []
for key, item in all_items.items():
    entity_list.append({"key": key, "type": item["type"], "data": item["data"]})


def question_yielder():
    yield from questions


def entity_yielder():
    yield from entity_list


question_dataset = Dataset.from_generator(question_yielder)
entity_dataset = Dataset.from_generator(entity_yielder)
full_dataset = DatasetDict(
    {"question_data": question_dataset, "entity_data": entity_dataset}
)
full_dataset.save_to_disk("temp/gen_tofu/dataset/")

# GENERATING SOME SPLITS (FROM THE DATASET OBJECT)

authors = full_dataset["entity_data"].filter(lambda row: row["type"] == "author")["key"]
publishers = full_dataset["entity_data"].filter(lambda row: row["type"] == "publisher")[
    "key"
]
books = full_dataset["entity_data"].filter(lambda row: row["type"] == "book")["key"]

author_forget = random.sample(authors, k=math.floor(len(authors) * 0.2))
publisher_forget = random.sample(publishers, k=1)
book_forget = random.sample(books, k=math.floor(len(books) * 0.3))

author_forget_split = question_dataset.filter(
    KeyChecker(author_forget, find_forget=True)
)
author_retain_split = question_dataset.filter(
    KeyChecker(author_forget, find_forget=False)
)
print(author_forget_split)
print(author_retain_split)

book_forget_split = question_dataset.filter(KeyChecker(book_forget, find_forget=True))
book_retain_split = question_dataset.filter(KeyChecker(book_forget, find_forget=False))
print(book_forget_split)
print(book_retain_split)

publisher_forget_split = question_dataset.filter(
    KeyChecker(publisher_forget, find_forget=True)
)

publisher_retain_split = question_dataset.filter(
    KeyChecker(publisher_forget, find_forget=False)
)
print(publisher_forget_split)
print(publisher_retain_split)
