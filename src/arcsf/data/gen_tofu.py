import csv
import json
import os
from uuid import uuid4

import networkx as nx
from pyvis.network import Network

from arcsf.data.generation.utils import (
    AuthorSampler,
    BookSampler,
    Formatter,
    PublisherSampler,
)

# DEFINING THE CONSTANTS

author_date_limits = ["01/01/1950", "01/01/2000"]
publisher_date_limits = ["01/01/1900", "01/01/2010"]
book_date_limits = ["01/01/1970", "01/01/2010"]

countries = ["Canada", "United Kingdom"]
country_map = {"Canada": "Canadian", "United Kingdom": "British"}
genres = ["Sci-Fi", "Crime", "History", "Architecture", "Motivational"]


country_items = {str(uuid4()): {"name": country} for country in countries}
genre_items = {str(uuid4()): {"name": genre} for genre in genres}


publishers = [
    "Albatross Press",
    "Clive & Sons",
    "Pelican Publishing",
    "Brilliant Books",
    "Notorious Novels",
    "Radical Writers",
]

publisher_country_distribution = {
    "options": list(country_items.keys()),
    "distribution": len(countries) * [3],
}

author_country_distribution = {
    "options": list(country_items.keys()),
    "distribution": len(countries) * [10],
}

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

publisher_items = {}

for publisher in publishers:
    pub_item = publisher_sampler.sample(publisher)
    publisher_items[pub_item["key"]] = pub_item

author_items = {}

for author_index in range(sum(author_country_distribution["distribution"])):
    author_item = author_sampler.sample()
    author_items[author_item["key"]] = author_item

# SAMPLE BOOKS

book_publisher_distribution = {
    "options": list(publisher_items.values()),
    "distribution": len(publishers) * [10],
}

book_author_distribution = {
    "options": list(author_items.values()),
    "distribution": len(author_items) * [3],
}

book_sampler = BookSampler(
    publisher_dist=book_publisher_distribution,
    author_dist=book_author_distribution,
    date_limits=book_date_limits,
)

book_items = {}

for book_idx in range(sum(book_author_distribution["distribution"])):
    book_item = book_sampler.sample()
    book_items[book_item["key"]] = book_item


# COMBINE ALL ITEMS

all_items = (
    {key: {"type": "publisher", "data": item} for key, item in publisher_items.items()}
    | {key: {"type": "author", "data": item} for key, item in author_items.items()}
    | {key: {"type": "book", "data": item} for key, item in book_items.items()}
    | {key: {"type": "country", "data": item} for key, item in country_items.items()}
    | {key: {"type": "genre", "data": item} for key, item in genre_items.items()}
)

# CREATE CONNECTION LIST

formatter = Formatter(all_items, country_map)

connections = []

for key in list(all_items.keys()):
    for connection in formatter.get_connections(key):
        connections.append(connection)

# SAVE ITEMS + CONNECTIONS

os.makedirs("temp/gen_tofu/", exist_ok=True)
with open("temp/gen_tofu/all_items.json", "w") as item_file:
    json.dump(all_items, item_file, indent=2)

with open("temp/gen_tofu/all_connections.csv", "w") as connection_file:
    file_writer = csv.writer(connection_file)
    for connection in connections:
        file_writer.writerow(connection)


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
net = Network(notebook=True, cdn_resources="in_line")
net.from_nx(graph)
net.save_graph("temp/gen_tofu/entire-graph.html")
