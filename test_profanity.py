import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from profanity import is_profane, detect

def test_flags_profanity():
    assert is_profane("fuck") and is_profane("Shit,") and is_profane("f***ing")

def test_clean_words_pass():
    for w in ["love", "night", "class", "assist", "Scunthorpe"]:
        assert not is_profane(w)

def test_detect_returns_timestamps():
    words = [{"word": "hello", "start": 0, "end": 1}, {"word": "shit", "start": 1, "end": 1.4}]
    assert [w["start"] for w in detect(words)] == [1]
