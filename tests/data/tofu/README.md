# Test data

## Dummy dataset

To create the dummy data run `python make_dummy_tofu.py`. This creates a small dataset of questions and
answers about synthetic authors, e.g. question: "what is sally synthetic favourite colour?", answer: "red".
Everything is lowercase and kept to a very restricted template/vocabulary to minimise the size of the dummy
tokenizer and model. It's saved to the `dummy_tofu_data` directory.

## Dummy model

To create the dummy tokenizer and model run `python make_base_model.py` (after creating the dummy dataset). This
will create and save a small GPT2-style tokenizer and model (single attention head, one attention block,
embedding dimension of 20, max sequence length of 40), fine-tuned on the dummy dataset. The dummy model is
good enough to be able to auto-complete sentences from the training data but not much more. It's saved to
the `dummy_base_gpt2` directory.

## Dummy unlearnt model

To create the unlearnt model run `python make_forget_model.py` (after creating the dataset and making the
base dummy model above). This further fine-tunes the dummy model with a simple forget objective to attempt
to unlearn the author sally synthetic. It's saved to the `dummy_forget_gpt2` directory.
