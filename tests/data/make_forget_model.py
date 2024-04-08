import shutil

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from arcsf.utils import seed_everything

seed = 42
seed_everything(seed)

tokenizer = AutoTokenizer.from_pretrained("dummy_base_gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize(sample):
    inputs = tokenizer(sample["text"])
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs


train_dataset = load_dataset("dummy_train_data", split="train")
forget_dataset = train_dataset.filter(lambda sample: sample["forget"]).map(tokenize)

model = AutoModelForCausalLM.from_pretrained("dummy_base_gpt2")


class SimpleForgetter(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        forget_loss = -1 * outputs.loss
        return (forget_loss, outputs) if return_outputs else forget_loss


tmp_dir = "tmp"
args = TrainingArguments(
    output_dir=tmp_dir,
    per_device_train_batch_size=1,
    num_train_epochs=17,
    learning_rate=5e-3,
    use_cpu=True,
    logging_steps=100,
    seed=seed,
)

trainer = SimpleForgetter(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=forget_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained("dummy_forget_gpt2")
shutil.rmtree(tmp_dir)
