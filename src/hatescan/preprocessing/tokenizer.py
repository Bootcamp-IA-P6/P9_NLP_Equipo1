import spacy
import nltk
from nltk.corpus import stopwords

from .cleaner import clean_text, extract_case_features


def _load_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


class TextTokenizer:
    """Tokenizer that cleans text, removes English stopwords, and lemmatizes with spaCy."""

    def __init__(self, model: str = "en_core_web_sm", preserve_case: bool = False):
        self.stopwords = _load_stopwords()
        self.nlp = spacy.load(model, disable=["parser", "ner"])
        self.preserve_case = preserve_case

    def tokenize(self, text: str) -> list[str]:
        cleaned = clean_text(text, preserve_case=self.preserve_case)
        if not cleaned:
            return []

        doc = self.nlp(cleaned)
        tokens: list[str] = []

        for token in doc:
            lemma = token.lemma_.lower().strip()
            if not lemma:
                continue
            if not token.is_alpha:
                continue
            if lemma in self.stopwords:
                continue
            tokens.append(lemma)

        return tokens

    def process(self, text: str) -> str:
        """Return processed text suitable for storage as text_processed."""
        return " ".join(self.tokenize(text))

    def get_features(self, text: str) -> dict:
        """Extract both text and case features for toxicity detection."""
        return {
            "text_processed": self.process(text),
            "case_features": extract_case_features(text)
        }
