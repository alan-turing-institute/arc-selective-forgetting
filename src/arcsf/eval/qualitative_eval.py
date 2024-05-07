import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from arcsf.data.data_module import EvalQADataset, get_data, qa_formatter_autoregression


def qualitative_eval(model, tokenizer, dataset, random_seed, **generate_kwargs):
    gen = torch.Generator().manual_seed(random_seed)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=gen)

    for batch_idx, (_, (question, answer)) in enumerate(data_loader):
        input_question = tokenizer.decode(question["input_ids"][0][0])
        target = tokenizer.decode(answer["input_ids"][0][0])
        output = model.generate(
            question["input_ids"][0],
            attention_mask=question["attention_mask"][0],
            **generate_kwargs,
        )
        generated_text = tokenizer.decode(
            output[0][len(question["input_ids"][0][0]) :], skip_special_tokens=True
        )
        print(f"\nQuestion index: {batch_idx}")
        print(f"Question: {input_question}")
        print(f"Generated: {generated_text}")
        print(f"Target: {target}")
        if batch_idx >= 5:
            break


if __name__ == "__main__":
    rand = 42
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model.config.pad_token_id = tokenizer.eos_token_id
    forget_data, retain_data = get_data(
        "tofu", "author", True, True, 0.2, 0.2, random_seed=rand
    )
    qa_formatter = qa_formatter_autoregression
    dataset = EvalQADataset(forget_data, tokenizer, qa_formatter, "standard")
    qualitative_eval(
        model,
        tokenizer,
        dataset,
        random_seed=rand,
        max_new_tokens=50,
        temperature=0.9,
        do_sample=True,
    )
