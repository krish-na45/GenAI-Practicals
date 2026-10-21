
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.eval()

context = """
A student must have a minimum attendance of 75 percent in each subject
to be eligible for the end semester examination. Students with attendance
between 65 and 75 percent may apply for condonation with a medical
certificate. The internal assessment carries 40 marks and the end
semester examination carries 60 marks. A student must score at least 40
percent overall to pass a course. The library remains open from 9 AM to
6 PM on all working days. Fee payment must be completed before the last
date announced by the accounts section, and a late fee of 500 rupees is
charged after the due date. Scholarship applications are submitted
through the college office within the first month of the semester.
"""


def answer(question):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start = int(outputs.start_logits.argmax())
    end = int(outputs.end_logits.argmax()) + 1

    answer_ids = inputs["input_ids"][0][start:end]

    return tokenizer.decode(
        answer_ids,
        skip_special_tokens=True
    )


questions = [
    "What is the minimum attendance required for the exam?",
    "How many marks does the internal assessment carry?",
    "What is the late fee after the due date?",
    "When is the library open?"
]

for question in questions:
    print("Q:", question)
    print("A:", answer(question))
    print()
