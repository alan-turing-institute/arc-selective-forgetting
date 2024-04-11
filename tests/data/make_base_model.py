"""
Creates and fine-tunes a small dummy GPT2-style tokenizer and model on the dummy dataset
"""

import os
import shutil

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

from arcsf.utils import seed_everything

seed = 42
seed_everything(seed)

# Create tokenizer
# Use ByteLevelBPETokenizer to create vocab.json and merges.txt files required as
# input to GPT2Tokenizer.
bpe_tokenizer = ByteLevelBPETokenizer()
bpe_tokenizer.train("vocab.txt", min_frequency=1)
bpe_path = "bpe_tokenizer"
os.makedirs(bpe_path, exist_ok=True)
bpe_tokenizer.save_model(bpe_path)
# Create and save GPT2Tokenizer
tokenizer = GPT2Tokenizer(
    f"{bpe_path}/vocab.json", f"{bpe_path}/merges.txt", model_max_length=40
)
shutil.rmtree(bpe_path)
gpt2_path = "dummy_base_gpt2"
tokenizer.save_pretrained(gpt2_path)


# Load and tokenizer dummy data
def tokenize(sample):
    inputs = tokenizer(sample["text"])
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


train_dataset = load_dataset("dummy_train_data", split="train").map(tokenize)
tokenizer.pad_token = tokenizer.eos_token

# Create and train small GPT2-architecutre model
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=40,
    n_embd=20,
    n_layer=1,
    n_head=1,
    bos_token_id=len(tokenizer) - 1,
    eos_token_id=len(tokenizer) - 1,
    seed=seed,
)
model = GPT2LMHeadModel(config)
model.init_weights()

tmp_dir = "tmp"
args = TrainingArguments(
    output_dir=tmp_dir,
    per_device_train_batch_size=1,
    num_train_epochs=2000,
    learning_rate=5e-3,
    use_cpu=True,
    logging_steps=1000,
    seed=seed,
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained(gpt2_path)
shutil.rmtree(tmp_dir)
