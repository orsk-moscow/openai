import pytest

from openaiWrapper.wrapper import OpenAIAPIWrapper


@pytest.fixture
def wrapper():
    return OpenAIAPIWrapper("gpt-3.5-turbo")


def test_init(wrapper):
    assert wrapper._model == "gpt-3.5-turbo"
    assert wrapper._price == OpenAIAPIWrapper._PRICES["gpt-3.5-turbo"]


def test_prices(wrapper):
    assert wrapper.prices == OpenAIAPIWrapper._PRICES


def test_price(wrapper):
    assert wrapper.price == OpenAIAPIWrapper._PRICES["gpt-3.5-turbo"]


def test_cost(wrapper, mocker):
    mocker.patch.object(OpenAIAPIWrapper, "_count_tokens")
    wrapper._counter = 10
    wrapper._cost()
    assert wrapper._money == 10 * OpenAIAPIWrapper._PRICES["gpt-3.5-turbo"] / 1000


def test_costs(wrapper):
    wrapper._money = 5
    assert wrapper.costs == 5


def test_preprocessing(wrapper):
    assert wrapper._preprocessing("Hello\nWorld") == "Hello World"
