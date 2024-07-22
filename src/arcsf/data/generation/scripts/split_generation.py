import random

import datasets

from arcsf.data.generation.utils import KeyChecker

# LOAD DATASET OBJECT
full_dataset = datasets.load_from_disk("temp/gen_tofu/new_dataset")
question_dataset = full_dataset["question_data"]
entity_dataset = full_dataset["entity_data"]

# GENERATING SOME SPLITS (FROM THE DATASET OBJECT)
authors = full_dataset["entity_data"].filter(lambda row: row["type"] == "author")["key"]
publishers = full_dataset["entity_data"].filter(lambda row: row["type"] == "publisher")[
    "key"
]
books = full_dataset["entity_data"].filter(lambda row: row["type"] == "book")["key"]

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
