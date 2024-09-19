import csv
import json
import os
import random
from uuid import uuid4

import networkx as nx
from datasets import Dataset
from pyvis.network import Network
from tqdm import tqdm

from arcsf.data.generation.basic_question_generation import NetworkQuestionGenerator
from arcsf.data.generation.gpt_generation import (
    IterativeGenerator,
    create_name_file,
    load_name_file,
    load_property_file,
)
from arcsf.data.generation.utils import (
    AuthorSampler,
    BookSampler,
    Formatter,
    PublisherSampler,
    flatten,
)

# RANDOM SEED
# random.seed(42)

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
book_date_limits = ["01/01/1970", "01/01/2020"]

with open("src/arcsf/data/generation/name_bank/countries.json", "r") as country_file:
    country_map = json.load(country_file)
countries = [key for key in country_map.keys()]

genres = [
    "Sci-Fi",
    "Mystery",
    "Romance",
    "Thriller",
    "Adventure",
    "Horror",
    "Western",
    "Crime",
    "Fantasy",
    "Detective",
]

# This allows us to represent these as entities in our graph, and their UUIDs will be
# used in other items to represent connections.
country_items = {str(uuid4()): {"name": country} for country in countries}
genre_items = {str(uuid4()): {"name": genre} for genre in genres}

# These allow us to give each book/publisher some distinguishing features
# whilst also tracking them in the dataset
property_items = {}
property_bank_dir = "src/arcsf/data/generation/property_bank/"

author_property_types = ["parent_relationship", "siblings", "education", "career"]
book_property_types = ["awards", "length", "sales"]
publisher_property_types = ["type"]

author_properties_dict = {}
book_properties_dict = {}
publisher_properties_dict = {}

for entity_type in ["author", "book", "publisher"]:
    for property_type in eval(f"{entity_type}_property_types"):
        property_names, weights = load_property_file(
            property_bank_dir, entity_type, property_type
        )
        property_keys = []
        for property_name in property_names:
            property_key = str(uuid4())
            property_items[property_key] = {
                "type": f"{entity_type}_{property_type}",
                "name": property_name,
            }
            property_keys.append(property_key)
        eval(f"{entity_type}_properties_dict")[property_type] = {
            "names": property_keys,
            "weights": weights,
        }

# Load publisher file

with open(
    "src/arcsf/data/generation/name_bank/publisher_names.csv", "r"
) as read_obj:  # read csv file as a list of lists
    csv_reader = csv.reader(
        read_obj
    )  # pass the file object to reader() to get the reader object
    publishers = flatten(list(csv_reader))


# 1 publisher per country
publisher_country_distribution = {
    "options": list(country_items.keys()),
    "distribution": len(countries) * [1],
}

# 5 authors per country
author_country_distribution = {
    "options": list(country_items.keys()),
    "distribution": len(countries) * [5],
}

# 20 authors per genre
author_genre_distribution = {
    "options": list(genre_items.keys()),
    "distribution": len(genres) * [20],
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
    for publisher_property_type, pub_prop_dict in publisher_properties_dict.items():
        pub_item[publisher_property_type] = random.choices(
            pub_prop_dict["names"], weights=pub_prop_dict["weights"]
        )[0]
    publisher_items[pub_item["key"]] = pub_item


for author_index in range(sum(author_country_distribution["distribution"])):
    author_item = author_sampler.sample()
    for author_property_type, author_prop_dict in author_properties_dict.items():
        author_item[author_property_type] = random.choices(
            author_prop_dict["names"], weights=author_prop_dict["weights"]
        )[0]
    author_items[author_item["key"]] = author_item

# SAMPLE BOOKS
# These need the publishers and authors generated first so we have UUIDs to sample from

# 20 books per publisher
book_publisher_distribution = {
    "options": list(publisher_items.values()),
    "distribution": len(publishers) * [20],
}
# ...also 4 books per author
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
    for book_property_type, book_prop_dict in book_properties_dict.items():
        book_item[book_property_type] = random.choices(
            book_prop_dict["names"], weights=book_prop_dict["weights"]
        )[0]
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
    | {
        key: {"type": item["type"], "data": {"name": item["name"]}}
        for key, item in property_items.items()
    }
)

# GENERATE NAMES FOR ENTITIES
name_bank_dir = "src/arcsf/data/generation/name_bank/"
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
            60,
        )

    for genre_key, author_count in tqdm(
        zip(
            author_genre_distribution["options"],
            author_genre_distribution["distribution"],
        )
    ):
        if all_items[genre_key]["data"]["name"]:
            create_name_file(
                name_bank_dir,
                "book",
                all_items[genre_key]["data"]["name"],
                100,
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

# now generate complex, longer form questions, if doing so
if GPT_GEN:
    iterative_generator = IterativeGenerator(all_items)
    n_gen = 2
    for book_profile in tqdm(
        list(book_items.values()),
        desc="Iterative Book Question Gen",
    ):
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
            questions.append(qa)

    # Generate fewer questions for authors/publishers than books
    n_gen = 1
    for author_profile in tqdm(
        list(author_items.values()),
        desc="Iterative Author Question Gen",
    ):
        keys = [
            author_profile["key"],
            author_profile["nationality"],
            author_profile["career"],
            author_profile["education"],
            author_profile["parent_relationship"],
            author_profile["siblings"],
        ]
        qa_pairs = []
        for key_index in [2, 4, 8]:
            iteration_keys = keys[:key_index]
            iteration_qa_pairs = iterative_generator.iterate_author_questions(
                author_profile, n_gen + 1, qa_pairs, iteration_keys
            )
            for iteration_qa in iteration_qa_pairs[: n_gen + 1]:
                iteration_row = {
                    "question": iteration_qa[0],
                    "answer": iteration_qa[1],
                    "keys": iteration_keys,
                }
                qa_pairs.append(iteration_row)

        for qa in qa_pairs:
            questions.append(qa)

    for publisher_profile in tqdm(
        list(publisher_items.values()),
        desc="Iterative Publisher Question Gen",
    ):
        keys = [
            publisher_profile["key"],
            publisher_profile["country"],
            publisher_profile["type"],
        ]
        qa_pairs = []
        for key_index in [2, 3]:
            iteration_keys = keys[:key_index]
            iteration_qa_pairs = iterative_generator.iterate_publisher_questions(
                publisher_profile, n_gen + 1, qa_pairs, iteration_keys
            )
            for iteration_qa in iteration_qa_pairs[: n_gen + 1]:
                iteration_row = {
                    "question": iteration_qa[0],
                    "answer": iteration_qa[1],
                    "keys": iteration_keys,
                }
                qa_pairs.append(iteration_row)

        for qa in qa_pairs:
            questions.append(qa)

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
    if item["type"] not in colour_map.keys():
        continue
    type = item["type"]
    graph.add_node(key, size=size_map[type], color=colour_map[type])

graph.add_edges_from(connections)
net = Network()
net.from_nx(graph)
net.repulsion(node_distance=250, central_gravity=0.5)
net.save_graph("temp/gen_tofu/graph.html")

# GENERATING DATSET

entity_list = []
types = []
for key, item in all_items.items():
    n_questions = 0
    entity_list.append({"key": key, "type": item["type"], "data": item["data"]})
    if item["type"] not in types:
        types.append(item["type"])
        for question in questions:
            for q_key in question["keys"]:
                if key == q_key:
                    # print(key)
                    # print(question["keys"])
                    # print("match")
                    n_questions += 1
        print(f"Type:{item['type']}    Number Questions: {n_questions}")


# GENERATOR FOR CREATING DATASET
def question_yielder():
    yield from questions


def entity_yielder():
    yield from entity_list


full_dataset = Dataset.from_generator(question_yielder)
full_dataset.save_to_disk("temp/gen_tofu/dataset/")
