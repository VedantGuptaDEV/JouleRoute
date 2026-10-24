"""Features for the logistic-regression baseline router.

Handcrafted signals (length, digits, operators, code-like tokens, question words) plus
hashed word 1-2-grams and char 3-5-grams. Everything is stateless except the scaler,
so the fitted pipeline is small and fast.
"""
import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

N_HASH_FEATURES = 2 ** 18

OPERATOR_RE = re.compile(r"[+\-*/=%^<>]")
CODE_TOKENS = ["def ", "return", "{", "}", "()", ";", "import ", "class ", "==", "python", "function"]
QUESTION_WORDS = ["why", "how", "explain", "prove", "calculate", "summarize", "write", "what", "who", "fix"]

HANDCRAFTED_NAMES = (
    ["chars", "words", "digits", "operators", "newlines"]
    + [f"code:{t.strip()}" for t in CODE_TOKENS]
    + [f"qword:{w}" for w in QUESTION_WORDS]
)


def handcrafted(texts) -> csr_matrix:
    rows = []
    for t in texts:
        low = t.lower()
        words = re.findall(r"\w+", low)
        rows.append(
            [
                np.log1p(len(t)),
                np.log1p(len(words)),
                np.log1p(sum(ch.isdigit() for ch in t)),
                np.log1p(len(OPERATOR_RE.findall(t))),
                np.log1p(t.count("\n")),
            ]
            + [float(tok in low) for tok in CODE_TOKENS]
            + [float(w in words) for w in QUESTION_WORDS]
        )
    return csr_matrix(np.asarray(rows, dtype=np.float64))


def build_featurizer():
    return make_pipeline(
        FeatureUnion([
            ("handcrafted", FunctionTransformer(handcrafted, accept_sparse=True)),
            ("word_ngrams", HashingVectorizer(ngram_range=(1, 2), n_features=N_HASH_FEATURES,
                                              alternate_sign=False, norm="l2")),
            ("char_ngrams", HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                              n_features=N_HASH_FEATURES, alternate_sign=False, norm="l2")),
        ]),
        MaxAbsScaler(),
    )
