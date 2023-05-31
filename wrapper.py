from typing import Optional

from tiktoken import Encoding
from tiktoken.model import encoding_for_model

from openai import ChatCompletion, Embedding


class OpenAIAPIWrapper:
    _PRICES: dict = {
        # prices in USD per 1k tokens
        # for details: https://openai.com/pricing#language-models
        "gpt-4": 0.03,  # chat
        "gpt-3.5-turbo": 0.002,  # chat
        "text-embedding-ada-002": 0.0004,  # embedder: 1536 points per vector
    }

    _available: str = "".join([f"'{model}'" for model in _PRICES.keys()])

    def __init__(self, model: str, limit: Optional[float] = None):
        f"""
        Initializes the OpenAIAPIWrapper with the specified model and optional limit.

        Args:
            model (str): The OpenAI model to use for token counting. Should be one of: {self._available}
            limit (Optional[float]): The optional limit on the number of tokens.

        Raises:
            ValueError: If the model is not in the allowed list or if the limit is not greater than 0.
        """
        if limit and limit <= 0:
            raise ValueError("Limit should be greater than 0")

        if model not in self._PRICES.keys():
            raise ValueError(
                f"Error model name: '{model}'. ",
                f"Should be one of: {self._available}",
            )

        self._model: str = model
        self._encoding: Encoding = encoding_for_model(model_name=model)
        self._price: float = self._PRICES[self._model]
        self._counter: int = 0
        self._money: float = 0.0
        self._limit: float = limit

    @property
    def prices(self) -> dict:
        """
        Returns the prices for all available models.

        Returns:
            dict: A dictionary mapping model names to their prices.
        """
        return self._PRICES

    @property
    def price(self) -> float:
        """
        Returns the price for the current model.

        Returns:
            float: The price for the current model.
        """
        return self._price

    def _cost(self):
        """
        Calculates the cost based on the number of tokens and the model price.
        """
        self._money += self._counter * self._price / 1_000
        self._counter = 0

    @property
    def costs(self) -> float:
        """
        Returns the total cost calculated so far.

        Returns:
            float: The total cost.
        """
        return self._money

    def _preprocessing(self, text: str) -> str:
        """
        Preprocesses the given text by replacing newlines with spaces.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        new = text.replace("\n", " ")
        return new

    def _embed(self, text: str) -> list[float]:
        """Embeds the given text using the OpenAI embeddings API.

        Code based above native methods:
        Example of usage: https://platform.openai.com/docs/guides/embeddings/use-cases
        API reference: https://platform.openai.com/docs/api-reference/embeddings/create

        Args:
            text (str): The input text to be embedded.

        Returns:
            list[float]: The embedding vector for the input text.

        Raises:
            ValueError: If the current model is not suitable for embedding operation.
        """
        if self._model != "text-embedding-ada-002":
            raise ValueError("The invocation of the 'embed' method for a model not prepared for such an operation")

        text = self._preprocessing(text)
        self._count_tokens(text)

        response = Embedding.create(input=[text], model=self._model)
        vector = response["data"][0]["embedding"]

        return vector

    def embed(self, text: str) -> list[float]:
        """
        Embeds the given text using the OpenAI embeddings API.

        Args:
            text (str): The input text to be embedded.

        Returns:
            list[float]: The embedding vector for the input text.
        """
        return self._embed(text)

    def chat_init(self, setup: str, text: str = "") -> str:
        """
        Initializes a chat with the OpenAI API.

        Args:
            setup (str): The setup message to be used for initializing the chat.
            text (str): The user message to be used for initializing the chat.

        Returns:
            str: The response from the API.
        """
        return self._chat_init(setup, text)

    def chat_continue(self, text: str) -> str:
        """
        Continues a chat with the OpenAI API.

        Args:
            text (str): The user message to be used for continuing the chat.

        Returns:
            str: The response from the API.
        """
        return self._chat_continue(text)

    def _chat_init(self, setup: str, text: str = "") -> str:
        """Initializes a chat with the OpenAI API.

        Example of usage: https://platform.openai.com/docs/guides/chat/introduction
        API reference: https://platform.openai.com/docs/api-reference/chat/create

        Args:
            setup (str): The setup message to be used for initializing the chat.
            text (str): The user message to be used for initializing the chat.

        Returns:
            str: The response from the API.
        """
        response = ChatCompletion.create(
            model=self._model,
            messages=[
                # The system message helps set the behavior of the assistant. In the example above, the assistant was instructed with “You are a helpful assistant
                {"role": "system", "content": setup},
                {"role": "user", "content": text},
            ],
        )
        response_text = response["choices"][0]["message"]["content"]
        return response_text

    def _chat_continue(self, text: str) -> str:
        """
        Continues a chat with the OpenAI API.

        Example of usage: https://platform.openai.com/docs/guides/chat/introduction
        API reference: https://platform.openai.com/docs/api-reference/chat/create

        Args:
            text (str): The user message to be used for continuing the chat.

        Returns:
            str: The response from the API.
        """
        response = ChatCompletion.create(
            model=self._model,
            messages=[
                {"role": "user", "content": text},
            ],
        )
        response_text = response["choices"][0]["message"]["content"]
        return response_text

    def _tokenize(self, text: str) -> list[int]:
        """
        Tokenizes the given text using the model's tokenizer.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: The tokens extracted from the text.
        """
        self._tokens = self._encoding.encode(text)

    def _count_tokens(self, text: str) -> int:
        """
        Counts the tokens in the given text using the model's tokenizer.

        Args:
            text (str): The text from which to count tokens.

        Returns:
            int: The count of tokens extracted from the text.
        """
        self._tokenize(text)
        self._token_count = len(self._tokens)
        self._counter += self._token_count

    def count(self, text: str) -> int:
        """
        Public method to count the tokens in the given text using the model's tokenizer.

        Args:
            text (str): The text from which to count tokens.

        Returns:
            int: The count of tokens extracted from the text.
        """
        text = self._preprocessing(text)
        self._count_tokens(text)
        return self._token_count


if __name__ == "__main__":
    text = "This is a sample sentence for token counting."
    text = "Этот текст взят просто для примера, чтобы подсчитать кол-во токенов."
    # gpt35turbo = OpenAIAPIWrapper("gpt-3.5-turbo")
    # num_tokens = gpt35turbo.count(text)
    ada = OpenAIAPIWrapper("text-embedding-ada-002")
    num_tokens = ada.count(text)

    # Print the number of tokens
    print(f"The number of tokens in the text is {num_tokens}")
    print(f"The number of symbols in the text is {len(text)}")
