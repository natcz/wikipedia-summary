from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
import torch
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_model


def return_sentence_embeddings(contents):
    model = SentenceTransformer("sdadas/st-polish-paraphrase-from-distilroberta")
    return model.encode(contents)


def top_articles_text_vec_similarity(
    content_chosen_article: str, contents: list[str]
) -> list:
    embeddings_all = return_sentence_embeddings(contents)
    embeddings_chosen = return_sentence_embeddings(content_chosen_article)
    cos_sim = util.cos_sim(embeddings_all, embeddings_chosen)
    return [
        article[1]
        for article in sorted(
            [[x[0], contents[i]] for i, x in enumerate(cos_sim.tolist())],
            key=lambda tup: tup[0],
            reverse=True,
        )
    ]


def top_articles_word_vec_similarity_fasttext(title: str) -> list:
    cap_path = datapath("data/wiki.pl.bin")
    fb_model = load_facebook_model(cap_path)
    return fb_model.wv.most_similar(title)


def similar_words_from_two_lists_bert(word_list1, word_list2):
    all_words = word_list1 + word_list2
    model_name = "dkleczek/bert-base-polish-uncased-v1"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    tokenized = tokenizer(all_words, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokenized)

    embeddings_list1 = outputs.last_hidden_state[: len(word_list1)].mean(dim=1)
    embeddings_list2 = outputs.last_hidden_state[len(word_list1) :].mean(dim=1)
    cosine_sim = torch.nn.functional.cosine_similarity(
        embeddings_list1, embeddings_list2
    )
    most_similar_words_from_list2 = [
        word_list2[index] for index in cosine_sim.argmax(axis=0)
    ]
    return most_similar_words_from_list2[:10]
