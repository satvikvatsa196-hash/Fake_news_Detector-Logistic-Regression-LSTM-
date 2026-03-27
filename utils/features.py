"""
Feature engineering for fake news detection.
Extracts linguistic, statistical, and NLP features from raw text.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ── Stop words (lightweight, no NLTK dependency) ──────────────────────────────
STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","shall","can",
    "to","of","in","for","on","with","at","by","from","up","about","into","through",
    "during","before","after","above","below","between","out","off","over","under",
    "again","further","then","once","here","there","when","where","why","how","all",
    "both","each","few","more","most","other","some","such","no","nor","not","only",
    "own","same","so","than","too","very","just","that","this","these","those",
    "it","its","they","them","their","we","our","you","your","he","she","his","her",
    "i","me","my","myself","yourself","himself","herself","itself","themselves",
}

# Credibility markers (signal real journalism)
CREDIBILITY_TERMS = [
    "according", "study", "research", "published", "journal", "university",
    "professor", "doctor", "scientist", "researcher", "analysis", "data",
    "evidence", "percent", "survey", "report", "confirmed", "officials",
    "spokesperson", "peer-reviewed", "findings", "institute", "laboratory",
    "clinical", "statistical", "methodology", "participants", "sample",
]

# Sensationalism markers (signal misinformation)
SENSATIONALISM_TERMS = [
    "shocking", "breaking", "secret", "exposed", "truth", "cover-up",
    "whistleblower", "silenced", "deleted", "wake up", "sheeple", "deep state",
    "mainstream media", "big pharma", "globalists", "crisis actors", "hoax",
    "fake news", "conspiracy", "they don't want", "share this", "before it's deleted",
    "anonymous source", "patriots", "red pill", "agenda", "censored",
]


def clean_text(text: str) -> str:
    """Lowercase, remove HTML/URLs/extra whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)         # URLs
    text = re.sub(r"<[^>]+>", " ", text)                 # HTML tags
    text = re.sub(r"[^a-z0-9\s!?]", " ", text)          # keep !? for features
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer without stop-word removal (for TF-IDF)."""
    return re.sub(r"[^a-z\s]", "", text.lower()).split()


def tokenize_no_stop(text: str) -> list[str]:
    """Tokenize and remove stop words."""
    return [w for w in tokenize(text) if w not in STOPWORDS and len(w) > 2]


# ── Linguistic feature extractor ─────────────────────────────────────────────
class LinguisticFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts hand-crafted linguistic features from raw text.
    Compatible with sklearn pipelines.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(text) for text in X], dtype=np.float32)

    def get_feature_names_out(self):
        return np.array([
            "exclamation_count", "question_count", "caps_ratio",
            "avg_sentence_len", "avg_word_len", "vocab_diversity",
            "credibility_score", "sensationalism_score",
            "num_numbers", "ellipsis_count", "url_count",
            "all_caps_words", "text_length", "word_count",
        ])

    def _extract(self, text: str) -> list:
        raw = str(text)
        lower = raw.lower()
        words = raw.split()
        n_words = max(len(words), 1)

        sentences = re.split(r"[.!?]+", raw)
        n_sentences = max(len([s for s in sentences if s.strip()]), 1)

        # Punctuation features
        exclamation_count = raw.count("!")
        question_count = raw.count("?")
        ellipsis_count = lower.count("...") + lower.count("…")

        # Casing features
        caps_ratio = sum(1 for c in raw if c.isupper()) / max(len(raw), 1)
        all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 2)

        # Lexical diversity
        clean_words = tokenize_no_stop(lower)
        avg_word_len = np.mean([len(w) for w in clean_words]) if clean_words else 0
        vocab_diversity = len(set(clean_words)) / max(len(clean_words), 1)

        # Sentence length
        avg_sentence_len = n_words / n_sentences

        # Semantic scores
        credibility_score = sum(1 for t in CREDIBILITY_TERMS if t in lower)
        sensationalism_score = sum(1 for t in SENSATIONALISM_TERMS if t in lower)

        # Other signals
        num_numbers = len(re.findall(r"\b\d+\.?\d*\b", raw))
        url_count = len(re.findall(r"http\S+|www\S+", raw))
        text_length = len(raw)

        return [
            exclamation_count, question_count, caps_ratio,
            avg_sentence_len, avg_word_len, vocab_diversity,
            credibility_score, sensationalism_score,
            num_numbers, ellipsis_count, url_count,
            all_caps_words, text_length, n_words,
        ]


def get_top_tfidf_words(vectorizer, n=20) -> tuple[list, list]:
    """Return most discriminating vocab words (real vs fake) by TF-IDF index."""
    names = vectorizer.get_feature_names_out()
    return names, names[:n]


def describe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick EDA summary for the dataset."""
    extractor = LinguisticFeatureExtractor()
    feats = extractor.transform(df["full_text"])
    feat_names = extractor.get_feature_names_out()
    feat_df = pd.DataFrame(feats, columns=feat_names)
    feat_df["label"] = df["label"].values
    return feat_df
