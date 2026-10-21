
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


templates = {
    "email": (
        "You are a professional office assistant. Write a formal "
        "and polite email to {to} about {topic}. Include a "
        "subject, greeting, body and closing. Keep it under 120 words."
    ),
    "report": (
        "You are a technical writer. Write a short report on {topic} "
        "with an introduction, three key points and a conclusion."
    ),
    "explain": (
        "You are a patient teacher. Explain {topic} in simple "
        "words for a beginner engineering student and give one example."
    )
}


def generate(prompt, max_new_tokens=250):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


def chatbot():
    print("AI Content Generation Chatbot (type 'exit' to quit)")

    while True:
        task = input("\nChoose task [email/report/explain]: ").strip().lower()

        if task == "exit":
            print("Chatbot closed.")
            break

        if task not in templates:
            print("Please enter email, report or explain.")
            continue

        topic = input("Enter the topic: ").strip()

        if not topic:
            print("Topic cannot be empty.")
            continue

        if task == "email":
            recipient = input("Recipient: ").strip()

            if not recipient:
                print("Recipient cannot be empty.")
                continue

            prompt = templates[task].format(
                to=recipient,
                topic=topic
            )
        else:
            prompt = templates[task].format(topic=topic)

        print("\nGenerated Content:\n")
        print(generate(prompt))


# Demonstration with fixed inputs
demo = [
    templates["email"].format(
        to="the Head of Department",
        topic="a request for leave on Friday"
    ),
    templates["report"].format(
        topic="the benefits of cloud computing"
    ),
    templates["explain"].format(
        topic="overfitting in machine learning"
    )
]

for prompt in demo:
    print("PROMPT:", prompt)
    print("OUTPUT:", generate(prompt))
    print()

# Uncomment the next line to run the interactive chatbot
# chatbot()
