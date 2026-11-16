import re
from pathlib import Path
from better_profanity import profanity as bp

bp.load_censor_words()

def load_extra(path="profanity_list.txt") -> set:
    p = Path(path)
    if not p.exists():
        return set()
    return {l.strip().lower() for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")}

EXTRA = load_extra()
SUFFIXES = ("ing", "ed", "er", "ers", "s", "in", "y")

def clean(word: str) -> str:
    return re.sub(r"[^a-z0-9*@$]", "", word.lower())

def is_profane(word: str, clf=None, threshold=0.5) -> bool:
    w = clean(word)
    if not w:
        return False
    if "*" in w:
        return True
    if bp.contains_profanity(w) or w in EXTRA:
        return True
    for s in SUFFIXES:
        if w.endswith(s) and w[:-len(s)] in EXTRA:
            return True
    if clf is not None:
        return clf.predict_proba([w])[0][1] >= threshold
    return False

def detect(words, clf=None):
    return [w for w in words if is_profane(w["word"], clf)]
