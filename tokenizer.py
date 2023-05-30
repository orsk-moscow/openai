from tiktoken.model import encoding_for_model


class OpenAITokenCounter:
    """
    A class to count the number of tokens in a given text string for a specific OpenAI model.

    Attributes:
        _model (str): The OpenAI model to use for token counting.
        _encoding: The tokenizer associated with the specified model.
        _tokens (list): The tokens extracted from the text.
        _token_count (int): The count of tokens extracted from the text.
    """

    def __init__(self, model):
        """
        Initializes the OpenAITokenCounter with the specified model.

        Args:
            model (str): The OpenAI model to use for token counting.
        """
        self._model = model
        self._encoding = encoding_for_model(model_name=model)

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

    def count(self, text: str) -> int:
        """
        Public method to count the tokens in the given text using the model's tokenizer.

        Args:
            text (str): The text from which to count tokens.

        Returns:
            int: The count of tokens extracted from the text.
        """
        self._count_tokens(text)
        return self._token_count


if __name__ == "__main__":
    # Instantiate the class with a model name
    token_counter = OpenAITokenCounter("gpt-3.5-turbo")
    text = "This is a sample sentence for token counting."
    num_tokens = token_counter.count(text)

    # Print the number of tokens
    print(f"The number of tokens in the text is {num_tokens}")
