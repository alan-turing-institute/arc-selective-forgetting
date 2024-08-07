import os
import time
from typing import Any

from torch import Tensor
from torch.utils.data import Dataset
from transformers import BatchEncoding, PreTrainedModel, Trainer
from transformers.trainer_utils import speed_metrics
from transformers.utils.generic import ModelOutput

from arcsf.eval.evaluate import EvaluateOutputs, Evaluator


class ARCSFTrainer(Trainer):
    """
    Modified version of the HuggingFace trainer that makes it possible to pass an
    instance of arcsf.eval.evaluate.Evaluator to evaluate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_save_dir = f"{self.args.output_dir}/../eval_checkpoints/"
        os.makedirs(self.eval_save_dir)

    def evaluate(
        self,
        eval_dataset: Evaluator | Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvaluateOutputs | dict[str, float]:
        """
        If eval_dataset is an Evaluator instance, evaluate using that. Otherwise, use
        the normal evaluate method from the HuggingFace trainer (parent class).
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if not isinstance(eval_dataset, Evaluator):
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        self.model.eval()
        self._memory_tracker.start()
        start_time = time.time()

        eval_outputs = eval_dataset.evaluate()
        # create new directory
        eval_outputs.save(f"{self.eval_save_dir}/epoch_{self.state.epoch:03d}.json")

        metrics = eval_outputs.summary_metrics
        metrics.update(
            speed_metrics(metric_key_prefix, start_time, num_samples=len(eval_dataset))
        )
        metrics = _add_prefix(metrics, f"{metric_key_prefix}_")
        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self._memory_tracker.stop_and_update_metrics(metrics)

        return eval_outputs


class Forgetter(ARCSFTrainer):
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


def _add_prefix(dictionary: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Add a prefix to all keys in a dictionary."""
    return {f"{prefix}{key}": value for key, value in dictionary.items()}
