import torch
from rouge_score.rouge_scorer import RougeScorer
from scipy.stats import ks_2samp


def ks_test(forget, retain):
    return ks_2samp(forget, retain)


def eval_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    accuracy = torch.mean(torch.eq(preds, labels).float())
    return {"eval_accuracy": accuracy}


def conditional_probability(normalised_losses):
    probs = torch.exp(-1 * normalised_losses)
    cond_probs = probs.T / torch.sum(probs, dim=-1)
    return cond_probs


def eval_rouge_recall(gen_outputs, ground_truths, indices):
    scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores["rouge1"].recall
        rougeL_recall[idx] = rouge_scores["rougeL"].recall

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}
