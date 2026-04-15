import os
import re
import nltk

# Cherche les données NLTK dans le dossier du projet (inclus dans le repo)
_NLTK_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nltk_data")
if os.path.isdir(_NLTK_DATA):
    nltk.data.path.insert(0, _NLTK_DATA)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_tweet(text: str) -> str:
    """Clean raw tweet text: remove mentions, URLs, special chars, numbers."""
    if not text:
        return ""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str, remove_stopwords: bool = True, method: str = "lemmatize") -> str:
    """Full preprocessing: clean, tokenize, remove stopwords, lemmatize or stem.

    Parameters
    ----------
    text : str
        Raw tweet text.
    remove_stopwords : bool
        Whether to remove English stopwords.
    method : str
        "lemmatize" (default) or "stem" (Porter stemmer).
    """
    if not text:
        return ""
    text = clean_tweet(text)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _stop_words]
    if method == "stem":
        tokens = [_stemmer.stem(t) for t in tokens]
    else:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)
