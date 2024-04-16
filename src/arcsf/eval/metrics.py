import torch
from rouge_score.rouge_scorer import RougeScorer
from torch.nn import Softmax

_softmax = Softmax(dim=-1)


def eval_accuracy(logits, labels):
    preds = logits.argmax(dim=-1)
    accuracy = torch.mean(torch.eq(preds, labels).float())
    return {"eval_accuracy": accuracy}


def eval_probability(logits):
    probs = _softmax(logits)
    eval_prob = probs[0] / torch.sum(probs)
    return {"eval_prob": eval_prob.item()}


def eval_rouge_recall(gen_outputs, ground_truths, indices):
    scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores["rouge1"].recall
        rougeL_recall[idx] = rouge_scores["rougeL"].recall

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}
