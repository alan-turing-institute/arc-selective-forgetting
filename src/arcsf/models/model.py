import logging

import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def load_maybe_peft_model(
    model_path_or_id: str, merge: bool = True, **model_kwargs
) -> PreTrainedModel | PeftModel:
    """
    Load either a PeftModel or a standard HuggingFace model from a given path or ID.

    Args:
        model_path_or_id: Path to a model or the model ID on HuggingFace.
        merge: If a PEFT model, whether to merge the adapter into the model before
            returning.
        **model_kwargs: Additional kwargs passed to initialise the model.

    Returns:
        PreTrainedModel if a normal base model or a merged PEFT model (with merge=True),
        otherwise a PEFTModel.
    """
    # Saved PEFT models should contain an adapter_config.json file, if loading with PEFT
    # fails due to this file not being found, then load as a standard model.
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path_or_id, **model_kwargs, torch_dtype="auto"
        )
        if merge:
            model = model.merge_and_unload()
    except ValueError as err:
        if "Can't find 'adapter_config.json'" not in str(err):
            raise err
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_id, **model_kwargs, torch_dtype="auto"
        )
    return model


def load_model_and_tokenizer(
    model_id: str,
    peft_kwargs: dict | None = None,
    add_padding_token: bool = False,
    **model_kwargs,
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizer]:
    """Function which given a model id and kwargs loads a HuggingFace model and
    associated tokenizer.

    Args:
        model_id: ID on huggingface of the model in question.
        peft_kwargs: Optional dictionary of kwargs to initialise a LoRA config which
                     is then used to make the model a PeftModel if LoRA fine-tuning
                     is desired.
        add_padding_token: Optional argument. If True and no padding token is present,
                           then a padding token will be added to the model.
        **model_kwargs: Additional kwargs passed to initialise the model.

    Returns:
        A tuple containing the model and the tokenizer.
    """

    # Initialise
    add_token_to_model = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Optionally add padding token
    if tokenizer.pad_token is None:
        if add_padding_token:
            logger.info("Adding pad token tok tokenizer and model.")
            tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
            add_token_to_model = True
        else:
            logger.warning("Defaulting to using EOS token as padding token.")
            tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load Model
    model = load_maybe_peft_model(model_id, merge=True, **model_kwargs)

    # If padding token added, add to model too
    if add_token_to_model:
        with torch.no_grad():
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    # Optionally convert model to peft model - assume LoRA for now
    if peft_kwargs is not None:
        logger.info(f"Converting to PEFT model with LoRA kwargs {peft_kwargs}")
        peft_config = LoraConfig(**peft_kwargs, task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)

    logger.info(f"Loaded model: {type(model)}\n{model}")
    logger.info(f"Model device: {model.device}")
    logger.info(f"Model dtype: {model.dtype}")

    # Return
    return model, tokenizer
