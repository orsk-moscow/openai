# OpenAI API Interaction

This repository contains Python scripts designed to interact with the OpenAI API. Utilizing the powerful language model developed by OpenAI, the scripts can generate general-purpose text embeddings, human-like text, answer questions, provide explanations, and much more.

## Features

- **Token Counter**: Estimate the cost of your API call by counting the number of tokens in your prompt using OpenAI's tiktoken Python library. 
 
## WARNING 

The pricing is limited and only applicable to a few selected models, particularly those required for downstream tasks. The figures provided are preliminary estimates intended to give you a ballpark idea of OpenAI API costs. 

Please use these figures responsibly and solely at your own discretion. They should be treated as a starting point for cost considerations, not as definitive charges. Always check the current rates directly from the OpenAI API for accurate pricing information.

## Getting Started

Clone this repository to your local machine and install the required packages.

```bash
git clone https://github.com/orsk-moscow/openai.git
cd openai
pip install -r requirements.txt
```

To interact with the OpenAI API, you need to provide your API key. The key can be found in the OpenAI dashboard.

```bash
touch .env && echo "OPENAI_API_KEY=<your-api-key>" >> .env
```

## Usage

You can interact with the OpenAI API using the provided Python scripts. Here is an example of how to use the token counter:

```bash
python token_counter.py "your-prompt-here"
```

```python
if __name__ == "__main__":
    # Instantiate the class with a model name
    token_counter = OpenAITokenCounter('gpt-3')

    # Use the class method to count the tokens in a text
    text = "This is a sample sentence for token counting."
    num_tokens = token_counter.count(text)

    # Print the number of tokens
    print(f"The number of tokens in the text is {num_tokens}")
```

## License

This project is licensed under the terms of the MIT license.
