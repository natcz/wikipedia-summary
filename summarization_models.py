from summa.summarizer import summarize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def textrank(text: str) -> str:
    return summarize(text, language="polish")


def bart(text: str, max_len: int = 150, min_len: int = 10) -> str:
    # model downloaded from: https://github.com/sdadas/polish-nlp-resources/releases/download/bart-base/bart_base_transformers.zip
    model_name = "data/bart_base_transformers"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = summarizer.generate(
        inputs.input_ids, max_length=max_len, min_length=min_len, do_sample=False
    )
    out = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return out
