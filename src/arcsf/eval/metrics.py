import torch
from rouge_score.rouge_scorer import RougeScorer
from scipy.stats import ks_2samp


def ks_test(forget, retain):
    return ks_2samp(forget, retain).pvalue


def eval_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    accuracy = torch.mean(torch.eq(preds, labels).float())
    return {"eval_accuracy": accuracy}


def conditional_probability(logit_normalised_losses):
    probs = torch.exp(-1 * logit_normalised_losses)
    cond_probs = probs.T / torch.sum(probs, dim=-1)
    return {"conditional_probs": cond_probs}


def eval_rouge_recall(gen_output, ground_truth):
    scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(ground_truth, gen_output)
    rouge1_recall = rouge_scores["rouge1"].recall
    rougeL_recall = rouge_scores["rougeL"].recall

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def truth_ratio(normalised_losses):
    numerator = torch.mean(
        conditional_probability(normalised_losses)["conditional_probs"][1:], dim=0
    )
    denominator = conditional_probability(normalised_losses)["conditional_probs"][0, :]
    return numerator / denominator
