import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# models = openai.Model.list()
# models
model_name = "gpt-3.5-turbo-0301"


def generate_response(prompt):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            # The system message helps set the behavior of the assistant. In the example above, the assistant was instructed with “You are a helpful assistant
            {"role": "system", "content": "You are an experienced musical editor who crafted a lot of playlists."},
            {"role": "user", "content": prompt},
        ],
    )
    return response["choices"][0]["message"]["content"]


# Get user input and generate a response
while True:
    prompt = input("Enter your question: ")
    if prompt.lower() == "quit":
        break
    answer = generate_response(prompt)
    print(answer)
