import json
import random

import datasets

from arcsf.data.generation.utils import KeyChecker

# LOAD DATASET OBJECT
question_dataset = datasets.load_from_disk("temp/gen_tofu/dataset")

with open("temp/gen_tofu/all_items.json", "r") as item_file:
    all_items = json.load(item_file)


# GENERATING SOME SPLITS FROM THE DICTIONARY OBJECT
authors = []
publishers = []
books = []

for key, item in all_items.items():
    if item["type"] == "author":
        authors.append(key)
    elif item["type"] == "publisher":
        publishers.append(key)
    elif item["type"] == "book":
        books.append(key)

# RANDOMLY REMOVE ONE OF EACH
author_forget = random.sample(authors, k=1)
publisher_forget = random.sample(publishers, k=1)
book_forget = random.sample(books, k=1)

# BOOK SPLIT
book_forget_split = question_dataset.filter(KeyChecker(book_forget, find_forget=True))
book_retain_split = question_dataset.filter(KeyChecker(book_forget, find_forget=False))
print(book_forget_split)
print(book_retain_split)

# AUTHOR SPLIT
author_forget_split = question_dataset.filter(
    KeyChecker(author_forget, find_forget=True)
)
author_retain_split = question_dataset.filter(
    KeyChecker(author_forget, find_forget=False)
)
print(author_forget_split)
print(author_retain_split)

# PUBLISHER SPLIT
publisher_forget_split = question_dataset.filter(
    KeyChecker(publisher_forget, find_forget=True)
)

publisher_retain_split = question_dataset.filter(
    KeyChecker(publisher_forget, find_forget=False)
)
print(publisher_forget_split)
print(publisher_retain_split)
