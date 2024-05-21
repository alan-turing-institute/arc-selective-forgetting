import torch
from rouge_score.rouge_scorer import RougeScorer
from scipy.stats import ks_2samp

# scorer so it doesn't need to be initialised on every call
scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def ks_test(forget: torch.Tensor, retain: torch.Tensor) -> float:
    """KS test figures 4-6 in the paper

    Args:
        forget : forget truth ratios
        retain : retain truth ratios

    Returns:
        p_value: returns the p_value of the ks_test for use in the forget quality
    """
    return ks_2samp(forget, retain).pvalue


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
) -> dict[torch.Tensor]:
    """
    Conditional probabilty defined in section 2.2.2 of the paper. Probabilities
    calculated using:
    p_i = exp(-1 * p_i) / (sum(exp(-1*p_j)) for j in N)

    Args:
        token_normalised_losses : losses on the answer normalised by number of tokens

    Returns:
        conditional_probs: conditional probabilities in the form of a dictionary
    """
    probs = torch.exp(-1 * token_normalised_losses)
    cond_probs = probs.T / torch.sum(probs, dim=-1)
    return {"conditional_probs": cond_probs}


def eval_rouge_recall(gen_output: str, ground_truth: str) -> dict[float, float]:
    """
    Abstraction of the RogueScorer class call, calculates the ROUGE score of the
    generated output. rougeL and rouge1 are outputted but the TOFU codebase only used
    rougeL.

    Args:
        gen_output : generated output
        ground_truth : ground truth string

    Returns:
        dict[float, float]: scores
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

    Returns:
        truth_ratio : the truth ratio
    """
    numerator = torch.mean(
        conditional_probability(normalised_losses)["conditional_probs"][1:], dim=0
    )
    denominator = conditional_probability(normalised_losses)["conditional_probs"][0, :]
    return numerator / denominator
