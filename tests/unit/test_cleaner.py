from src.hatescan.preprocessing.cleaner import clean_text, extract_case_features


def test_cleaner_removes_urls_mentions_hashtags():
    text = "Visit https://example.com and say hi @user #hello"
    assert clean_text(text) == "visit and say hi"


def test_cleaner_normalizes_spaces_and_lowercase():
    text = "  THIS   is   A   Test  "
    assert clean_text(text) == "this is a test"


def test_cleaner_returns_empty_for_empty_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_cleaner_keeps_already_clean_text():
    text = "clean text only"
    assert clean_text(text) == "clean text only"


def test_cleaner_removes_www_urls():
    text = "Go to www.example.com now"
    assert clean_text(text) == "go to now"


def test_cleaner_can_preserve_case():
    text = "THIS IS SHOUTING"
    assert clean_text(text, preserve_case=True) == "THIS IS SHOUTING"
    assert clean_text(text, preserve_case=False) == "this is shouting"


def test_extract_case_features():
    features = extract_case_features("THIS IS SHOUTING!!!")
    assert features["has_uppercase"] == True
    assert features["all_caps_ratio"] > 0
    assert features["exclamation_count"] == 3

    features_normal = extract_case_features("this is normal text")
    assert features_normal["has_uppercase"] == False
    assert features_normal["all_caps_ratio"] == 0.0
