import evaluate
from rouge_score import rouge_scorer
import pandas as pd
from pathlib import Path
from datetime import datetime

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, candidate)
    return scores


def calculate_bleu(reference, candidate):
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=candidate, references=reference)
    return results["bleu"]


def evaluate_metrics(reference, candidate):
    if candidate is None:
        return 0, 0
    bleu_score = calculate_bleu([reference], [" ".join(candidate)])
    rogue_score = calculate_rouge(reference, " ".join(candidate))
    return bleu_score, rogue_score


def save_metrics(bleu_scores, rouge_scores, how_many, dir_name):
    (
        rouge1_precision,
        rouge1_recall,
        rouge1_fmeasure,
        rouge2_precision,
        rouge2_recall,
        rouge2_fmeasure,
        rougel_precision,
        rougel_recall,
        rougel_fmeasure,
    ) = rouge_scores

    avg_bleu_score = bleu_scores/ how_many 
    avg_rouge_score = {
        "rouge-1": {
            "precision": rouge1_precision / how_many,
            "recall": rouge1_recall / how_many,
            "f1": rouge1_fmeasure / how_many,
        },
        "rouge-2": {
            "precision": rouge2_precision / how_many,
            "recall": rouge2_recall / how_many,
            "f1": rouge2_fmeasure / how_many,
        },
        "rouge-l": {
            "precision": rougel_precision / how_many,
            "recall": rougel_recall / how_many,
            "f1": rougel_fmeasure / how_many,
        },
    }
    
    scores = [['bleu', avg_bleu_score], ['rouge', avg_rouge_score]] 
    scores_df = pd.DataFrame(scores, columns=['metric', 'score']) 
    datetime_name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + ".csv"
    file_path = Path(dir_name) / datetime_name
    scores_df.to_csv(file_path, scores_df)

