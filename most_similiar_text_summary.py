from evaluation import evaluate_metrics, save_metrics
from constant import ARTICLES_DF as articles_df
from similarity_utils import top_articles_text_vec_similarity
from summarization_models import textrank, bart


def most_similar_text_summary(
    summarization_func: callable, summ_func_name: str
) -> None:
    bleu_scores = 0
    rouge1_precision = 0
    rouge2_precision = 0
    rougel_precision = 0

    rouge1_recall = 0
    rouge2_recall = 0
    rougel_recall = 0

    rouge1_fmeasure = 0
    rouge2_fmeasure = 0
    rougel_fmeasure = 0

    count = 0

    for article_title in articles_df.title.values.tolist():
        article_content = articles_df[
            articles_df.title == article_title
        ].lemmatized.values[0]
        corpus = [
            article
            for article in articles_df.lemmatized.values.tolist()
            if article != article_content
        ]
        top_similar_articles = top_articles_text_vec_similarity(
            article_content, corpus
        )[:10]
        bleu_score, rouge_score = evaluate_metrics(
            article_content, summarization_func(top_similar_articles)
        )

        bleu_scores += bleu_score

        rouge1_precision += rouge_score["rouge1"][0]
        rouge2_precision += rouge_score["rouge2"][0]
        rougel_precision += rouge_score["rougeL"][0]

        rouge1_recall += rouge_score["rouge1"][1]
        rouge2_recall += rouge_score["rouge2"][1]
        rougel_recall += rouge_score["rougeL"][1]

        rouge1_fmeasure += rouge_score["rouge1"][2]
        rouge2_fmeasure += rouge_score["rouge2"][2]
        rougel_fmeasure += rouge_score["rougeL"][2]

        count += 1

    rouge_scores = (
        rouge1_precision,
        rouge1_recall,
        rouge1_fmeasure,
        rouge2_precision,
        rouge2_recall,
        rouge2_fmeasure,
        rougel_precision,
        rougel_recall,
        rougel_fmeasure,
    )

    save_metrics(bleu_scores, rouge_scores, count, f"text_similarity_{summ_func_name}")


most_similar_text_summary(bart, "bart")
most_similar_text_summary(textrank, "textrank")
