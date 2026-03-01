from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


_DEFAULT_GENERIC_PHRASES_ID = [
    # Indonesian marketplace cliches
    "sesuai deskripsi",
    "barang bagus",
    "barang sangat bagus",
    "mantap",
    "recommended",
    "rekomen",
    "pengiriman cepat",
    "packing rapi",
    "penjual ramah",
    "terima kasih",
    "produk original",
    "harga murah",
    "ok",
    "oke",
    "bagus",
    "puas",
    "suka",
    "sesuai",
]


_WORD_RE = re.compile(r"[\w']+", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\d")


def normalize_text(text: str) -> str:
    """Light normalization: lowercase + trim."""
    if text is None:
        return ""
    return str(text).strip().lower()


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return _WORD_RE.findall(text)


def type_token_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / float(len(tokens))


def digit_count(text: str) -> int:
    return len(_DIGIT_RE.findall(text or ""))


def generic_phrase_hits(text: str, phrases: Iterable[str]) -> int:
    t = normalize_text(text)
    hits = 0
    for p in phrases:
        p = normalize_text(p)
        if p and p in t:
            hits += 1
    return hits


@dataclass
class IndicatorConfig:
    generic_phrases: List[str]

    @staticmethod
    def default() -> "IndicatorConfig":
        return IndicatorConfig(generic_phrases=list(_DEFAULT_GENERIC_PHRASES_ID))


def extract_indicators(
    texts: List[str],
    cfg: Optional[IndicatorConfig] = None,
) -> np.ndarray:
    """Return numeric indicator matrix shape (n, k). Non-negative features.

    Features:
      0 length_chars
      1 length_tokens
      2 digit_count
      3 ttr (0..1)
      4 generic_hits
    """
    if cfg is None:
        cfg = IndicatorConfig.default()

    n = len(texts)
    out = np.zeros((n, 5), dtype=float)
    for i, tx in enumerate(texts):
        txn = normalize_text(tx)
        toks = tokenize(txn)
        out[i, 0] = len(txn)
        out[i, 1] = len(toks)
        out[i, 2] = digit_count(txn)
        out[i, 3] = type_token_ratio(toks)
        out[i, 4] = generic_phrase_hits(txn, cfg.generic_phrases)
    return out


def nearest_neighbor_similarity(
    texts: List[str],
    reference_texts: Optional[List[str]] = None,
    vectorizer: Optional[TfidfVectorizer] = None,
    max_features: int = 50000,
    ngram_range=(1, 2),
) -> np.ndarray:
    """Compute cosine similarity to nearest neighbor.

    - If reference_texts is None: compute similarity within texts (excluding self).
    - If reference_texts provided: compute similarity of each text to the most similar
      reference text.

    Returns shape (n,) with values in [0,1].

    Notes:
    - Intended for MVP/small batches. For very large corpora, use ANN indexing.
    """
    texts_n = [normalize_text(t) for t in texts]
    if reference_texts is None:
        ref_n = texts_n
    else:
        ref_n = [normalize_text(t) for t in reference_texts]

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=1,
        )
        vectorizer.fit(ref_n)

    X = vectorizer.transform(texts_n)
    R = vectorizer.transform(ref_n)

    # Cosine similarity for TF-IDF vectors is dot product because TF-IDF is L2-normalized
    # by default in scikit-learn.
    sims = (X @ R.T).tocsr()

    if reference_texts is None:
        # remove self-similarity by setting diagonal to 0
        # only valid if texts and ref are same length and aligned
        n = sims.shape[0]
        if sims.shape[0] == sims.shape[1]:
            sims.setdiag(0.0)
            sims.eliminate_zeros()

    # nearest neighbor max similarity per row
    nn = np.zeros((sims.shape[0],), dtype=float)
    for i in range(sims.shape[0]):
        row = sims.getrow(i)
        if row.nnz:
            nn[i] = row.data.max()
        else:
            nn[i] = 0.0
    return nn
