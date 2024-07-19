from torch import Tensor
from transformers import BatchEncoding, PreTrainedModel, Trainer
from transformers.utils.generic import ModelOutput

from arcsf.eval.evaluate import Evaluator


class Forgetter(Trainer):
    """
    Forgetter base class, which defines the interface for all forgetters. Forgetters are
    modified versions of the HuggingFace Trainer class that compute a loss function
    based on forget and optionally retain (or 'I don't know') inputs.

    Any eval_dataset input is expected to be a arcsf.eval.evaluate.Evaluator instance.
    Apart from parameters relating to when to run evaluation, all eval parameters set in
    the trainer will be ignored, only what's set in the Evaluator is used.

    See the documentation of the HuggingFace Trainer for more usage information.
    """

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: tuple[BatchEncoding, BatchEncoding],
        return_outputs: bool = False,
    ) -> Tensor | tuple[Tensor, ModelOutput]:
        """
        Compute the unlearning loss of the model on the forget and retain inputs.

        Args:
            model: The model to compute the loss of.
            inputs: Tuple of forget and either retain or IDK inputs, as returned by
                QAForgetDataset. All child classes of Forgetter should expect two inputs
                in this order.
            return_outputs: Whether to return the outputs of the model or just the loss.

        Returns:
            The unlearning loss of the model, or the loss and the outputs of the model
            if return_outputs is True.
        """
        raise NotImplementedError(
            "A Forgetter child class implementing compute_loss should be used"
        )

    def evaluate(
        self,
        eval_dataset: Evaluator | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """TODO - implement evaluation metrics after evaluation PR merged"""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_outputs = eval_dataset.evaluate()
        self.log(eval_outputs.summary_metrics)
        return eval_outputs
