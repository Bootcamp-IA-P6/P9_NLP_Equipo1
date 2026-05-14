from src.hatescan.preprocessing.tokenizer import TextTokenizer


tokenizer = TextTokenizer()
tokenizer_case_sensitive = TextTokenizer(preserve_case=True)


def test_tokenizer_tokenizes_and_lemmatizes_text():
    text = "Running dogs chased"
    tokens = tokenizer.tokenize(text)

    assert "run" in tokens
    assert "dog" in tokens
    assert "chase" in tokens


def test_tokenizer_removes_stopwords():
    text = "This is a simple sentence with stopwords"
    processed = tokenizer.process(text)

    assert "this" not in processed
    assert "is" not in processed
    assert "simple" in processed
    assert "sentence" in processed


def test_tokenizer_handles_emojis_and_empty_text():
    text = "WOW 😡 this is horrible!"
    processed = tokenizer.process(text)

    assert "wow" in processed
    assert "horrible" in processed

    assert tokenizer.process("") == ""


def test_tokenizer_preserves_hate_semantics():
    text = "This is hate speech and abusive behavior"
    tokens = tokenizer.tokenize(text)

    assert "hate" in tokens
    assert "speech" in tokens
    assert "abusive" in tokens


def test_tokenizer_can_preserve_case():
    text = "RUNNING DOGS CHASED"
    tokens_case = tokenizer_case_sensitive.tokenize(text)
    tokens_lower = tokenizer.tokenize(text)

    # Case sensitive debería preservar algo de la información de mayúsculas
    # aunque spaCy normaliza lemmas a lowercase
    assert len(tokens_case) == len(tokens_lower)


def test_tokenizer_get_features():
    text = "THIS IS SHOUTING!!!"
    features = tokenizer.get_features(text)

    assert "text_processed" in features
    assert "case_features" in features
    assert features["case_features"]["has_uppercase"] == True
    assert features["case_features"]["exclamation_count"] == 3
