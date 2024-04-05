from transformers import Trainer


class Forgetter(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        raise NotImplementedError()
