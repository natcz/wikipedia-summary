from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from evaluation import evaluate_metrics, save_metrics
from constant import ARTICLES_DF as articles_df
from constant import STOPLIST as stoplist
from summarization_models import textrank, bart

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=1000, stop_words=stoplist)
tfidf_matrix = tfidf_vectorizer.fit_transform(articles_df["lemmatized"])

num_clusters = 1000
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)
articles_df["cluster"] = kmeans.labels_
clustered_articles = articles_df.groupby("cluster")["lemmatized"]


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

    for cluster_id, grouped_articles in clustered_articles:
        reference_articles = [
            article for row_index, article in grouped_articles.items()
        ]

        for reference_article in reference_articles:
            concatenated_articles = [
                article
                for article in reference_articles
                if article != reference_article
            ]
            bleu_score, rouge_score = evaluate_metrics(
                reference_article, summarization_func(concatenated_articles)
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

    save_metrics(bleu_scores, rouge_scores, count, f"kmean_{summ_func_name}")


most_similar_text_summary(bart, "bart")
most_similar_text_summary(textrank, "textrank")
