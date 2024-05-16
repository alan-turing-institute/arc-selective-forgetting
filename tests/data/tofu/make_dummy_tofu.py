import os
import re

from datasets import Dataset

questions = [
    "what is {author} favourite colour?",
    "what genre does {author} write?",
    "who wrote {book}?",
]
authors = {
    "sally synthetic": ["red", "thriller", "suspense"],
    "frank fake": ["green", "fantasy", "magic"],
    "alice artificial": ["blue", "scifi", "space"],
}
idk_responses = ["dont know", "unclear"]
forget_authors = ["sally synthetic"]

dataset = {"question": [], "answer": [], "forget": []}
for author, answers in authors.items():
    color, genre, book = answers
    dataset["answer"] += [color, genre, author]
    for question in questions:
        dataset["question"].append(
            question.format(author=author, color=color, genre=genre, book=book)
        )
        if author in forget_authors:
            dataset["forget"].append(True)
        else:
            dataset["forget"].append(False)

dataset = Dataset.from_dict(dataset)
name = "dummy_tofu_data"
os.makedirs(name, exist_ok=True)
dataset.to_parquet(f"{name}/{name}.parquet")

# generate file of vocabulary words to use for initialising tokenizer
all_texts = " ".join(idk_responses) + " "
for qa in dataset:
    all_texts += qa["question"] + " " + qa["answer"] + " "
all_texts = re.sub(r"[^\w\s]", "", all_texts).strip()
vocab = set(all_texts.split())
with open("vocab.txt", "w") as f:
    f.write("\n".join(vocab))

with open("idk.txt", "w") as f:
    f.write("\n".join(idk_responses))
