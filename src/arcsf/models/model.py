import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def load_model_and_tokenizer(
    model_id: str,
    peft_kwargs: dict | None = None,
    **model_kwargs,
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizer]:

    # Initialise
    add_token_to_model = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Optionally add padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
        add_token_to_model = True

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )

    # If padding token added, add to model too
    if add_token_to_model:
        with torch.no_grad():
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    # Optionally convert model to peft model - assume LoRA for now
    if peft_kwargs is not None:
        peft_config = LoraConfig(**peft_kwargs, task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)

    # Return
    return model, tokenizer
