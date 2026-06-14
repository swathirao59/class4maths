from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=512
)


def generate_answer(prompt):

    result = generator(
        prompt,
        temperature=0.3,
        do_sample=True
    )

    return result[0]["generated_text"]
