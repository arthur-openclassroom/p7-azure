import pytest
from api.preprocessing import clean_tweet, preprocess_text


class TestCleanTweet:
    def test_removes_mentions(self):
        assert clean_tweet("@user hello") == "hello"

    def test_removes_urls(self):
        assert clean_tweet("check http://example.com out") == "check out"
        assert clean_tweet("check https://example.com out") == "check out"

    def test_removes_special_characters(self):
        assert clean_tweet("hello!! world??") == "hello world"

    def test_converts_to_lowercase(self):
        assert clean_tweet("Hello World") == "hello world"

    def test_removes_numbers(self):
        assert clean_tweet("flight 123 delayed") == "flight delayed"

    def test_removes_extra_whitespace(self):
        assert clean_tweet("hello   world") == "hello world"

    def test_handles_empty_string(self):
        assert clean_tweet("") == ""

    def test_combined_cleaning(self):
        tweet = "@airline Your flight #123 is LATE!! http://t.co/abc"
        result = clean_tweet(tweet)
        assert result == "your flight is late"


class TestPreprocessText:
    def test_tokenizes_text(self):
        result = preprocess_text("hello world")
        assert isinstance(result, str)

    def test_lemmatizes_words(self):
        result = preprocess_text("flying planes")
        assert "fly" in result or "plane" in result

    def test_handles_empty_string(self):
        result = preprocess_text("")
        assert result == ""
