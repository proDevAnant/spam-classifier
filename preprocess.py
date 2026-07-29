"""
preprocess.py
-------------
Text cleaning utilities for the Spam Classifier.

We deliberately avoid NLTK's stopwords/punkt downloads (they need internet
the first time they run) and instead use scikit-learn's built-in English
stopword list, so this works completely offline.
"""

import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = ENGLISH_STOP_WORDS


def clean_text(text: str) -> str:
    """
    Lowercase, remove URLs/numbers/punctuation, and strip stopwords.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove standalone numbers (keep words that contain letters)
    text = re.sub(r"\b\d+\b", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # remove stopwords
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]

    return " ".join(words)
