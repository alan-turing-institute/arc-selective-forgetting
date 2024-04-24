"""
Creates small dummy dataset with strings like
"what is the capital of france? the capital of france is paris"
"""

import os
import re

from datasets import Dataset

# create dummy strings
countries = ["france", "germany", "italy"]
cities = ["paris", "berlin", "rome"]
idk_countries = ["england"]
idk_cities = ["london"]
idk_responses = ["dont know", "unclear"]
forget_city = "paris"

texts = []
forget = []  # True / False (whether this will be in the forget split)
strings = [
    "what is the capital of {country}? the capital of {country} is {city}",
    "the capital of {country} is {city}",
]
for country, city in zip(countries, cities):
    for s in strings:
        texts.append(s.format(country=country, city=city))
        if city == forget_city:
            forget.append(True)
        else:
            forget.append(False)

for idk_country in idk_countries:
    for idk_response in idk_responses:
        texts.append(f"what is the capital of {idk_country}? {idk_response}")
        forget.append(False)

# convert to Dataset and save
dataset = Dataset.from_dict({"text": texts, "forget": forget})
name = "dummy_train_data"
os.makedirs(name, exist_ok=True)
dataset.to_parquet(f"{name}/{name}.parquet")

# generate file of vocabulary words to use for initialising tokenizer
all_texts = " ".join(texts)
all_texts = re.sub(r"[^\w\s]", "", all_texts)
vocab = set(all_texts.split())
with open("vocab.txt", "w") as f:
    f.write("\n".join(vocab))
