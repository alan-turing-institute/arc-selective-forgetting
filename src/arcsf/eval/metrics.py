import torch
from rouge_score.rouge_scorer import RougeScorer
from scipy.stats import ks_2samp
from torch.nn import CrossEntropyLoss

# scorer so it doesn't need to be initialised on every call
scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def ks_test(forget: torch.Tensor, retain: torch.Tensor, **kwargs) -> float:
    """KS test figures 4-6 in the paper

    Args:
        forget : forget truth ratios
        retain : retain truth ratios

    Returns:
        p_value: returns the p_value of the ks_test for use in the forget quality
    """
    return ks_2samp(forget, retain, **kwargs).pvalue


def ecdf(x):
    xs, _ = torch.sort(x)
    ys = torch.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def eval_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> dict[torch.Tensor]:
    """Gets the accuracy given logits, this isn't used in the paper.

    Args:
        logits : output logits
        labels : targets

    Returns:
         accuracy : accuracy metric in the form of a dictionary
    """
    preds = logits.argmax(dim=-1)
    accuracy = torch.mean(torch.eq(preds, labels).float())
    return {"eval_accuracy": accuracy}


def conditional_probability(
    token_normalised_losses: torch.Tensor,
) -> torch.Tensor:
    """
    Conditional probabilty defined in section 2.2.2 of the paper. Probabilities
    calculated using:
    p_i = exp(-1 * p_i) / (sum(exp(-1*p_j)) for j in N)

    Args:
        token_normalised_losses : losses on the answer normalised by number of tokens
            shape: n_samples x n_perturbed

    Returns:
        conditional_probs: conditional probabilities in the form of a dictionary
    """
    # shape : n_samples x n_perturbed + 1
    probs = torch.exp(-1 * token_normalised_losses)
    sum = torch.sum(probs, dim=-1)  # shape : n_samples
    # transpose here ensures that the n_samples dimension is in the right place
    cond_probs = probs.T / sum  # shape : n_perturbed + 1 x n_samples
    return cond_probs


def eval_rouge_recall(gen_output: str, ground_truth: str) -> dict[str, float]:
    """
    Abstraction of the RogueScorer class call, calculates the ROUGE score of the
    generated output. rougeL and rouge1 are outputted but the TOFU codebase only used
    rougeL.

    Args:
        gen_output : generated output
        ground_truth : ground truth string

    Returns:
        ROUGE scores
    """
    rouge_scores = scorer.score(ground_truth, gen_output)
    rouge1_recall = rouge_scores["rouge1"].recall
    rougeL_recall = rouge_scores["rougeL"].recall

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def truth_ratio(normalised_losses: torch.Tensor) -> torch.Tensor:
    """
    Calculates the truth ratio given normalised losses. Equation (1) in the paper.
    The `conditional_probablity` function can be used here.

    Args:
        normalised_losses : Token length normalised losses.
            shape: n_samples x number_perturbed

    Returns:
        truth_ratio : the truth ratio
    """
    # getting conditional probability of the losses, this does perform a transpose
    # which doesn't affect the output values, but also isn't strictly necessary.

    # cond_probs shape: n_perturbed + 1 x n_samples
    cond_probs = conditional_probability(normalised_losses)
    numerator = torch.mean(cond_probs[1:, :], dim=0)  # shape: n_samples
    denominator = cond_probs[0, :]  # shape: n_samples
    return numerator / denominator  # shape: n_samples


loss_function = CrossEntropyLoss(ignore_index=-100, reduction="none")


def get_loss(output_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute loss along a batch from the evaluation script

    Args:
        output_logits: output logits from model (batch_size x sequence_length x
            vocab_size)
        labels: labels (batch_size x sequence_length)

    Returns:
        Normalised loss for each sample in the batch
    """
    # shape: batch_size x (sequence_length-1) x vocab_size
    output_logits = output_logits[..., :-1, :].contiguous()

    # shape : batch_size x (sequence_length - 1)
    shifted_labels = labels[..., 1:].contiguous()
    # output_logits.transpose(-1, -2) shape: batch_size x vocab x (sequence_length - 1)
    # loss shape: batch_size
    loss = loss_function(output_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    target_len = torch.sum(labels != -100, dim=-1)  # length of tokens in target
    loss_normalised = loss / target_len  # normalised loss shape: batch_size
    return loss_normalised
