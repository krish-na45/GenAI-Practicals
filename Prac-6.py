
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

article = """
The city council has approved a new metro rail project that will
connect the eastern suburbs with the central business district.
The project is expected to cost nearly 8,000 crore rupees and will
include 18 stations along a 24 kilometre route. Officials said the
metro will reduce travel time from almost 90 minutes to about
35 minutes and will help cut traffic congestion on the main roads.
Construction is scheduled to begin early next year and the first
phase is expected to open within four years. Environmental groups
welcomed the plan but asked the council to protect existing trees
and green areas during construction. The council has promised to
plant two trees for every one that is removed and to publish
quarterly progress reports for residents.
"""


def summarize(text, max_len=60, min_len=25):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    with torch.no_grad():
        ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=max_len,
            min_length=min_len,
            early_stopping=True
        )

    return tokenizer.decode(
        ids[0],
        skip_special_tokens=True
    )


summary = summarize(article)

print("Original length:", len(article.split()), "words")
print("Summary length:", len(summary.split()), "words")

print("\nSummary:")
print(summary)
