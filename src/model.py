from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from .features import IndicatorConfig, extract_indicators, nearest_neighbor_similarity, normalize_text


@dataclass
class SavedModel:
    vectorizer: TfidfVectorizer
    clf: MultinomialNB
    indicator_cfg: IndicatorConfig
    num_feature_max: np.ndarray  # for simple scaling


def _scale_nonneg(X: np.ndarray, feature_max: np.ndarray) -> np.ndarray:
    """Scale non-negative numeric features by per-column max."""
    feature_max = np.where(feature_max <= 0, 1.0, feature_max)
    return X / feature_max


def build_features(
    texts: List[str],
    vectorizer: TfidfVectorizer,
    indicator_cfg: IndicatorConfig,
    num_feature_max: Optional[np.ndarray] = None,
    similarity_reference_texts: Optional[List[str]] = None,
    similarity_vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build combined sparse features = [TFIDF | numeric_indicators(+nn_similarity)].

    Returns:
      X: sparse csr matrix
      num_all: raw numeric features (shape: n, k)
      nn: nearest-neighbor similarity vector (shape: n,)
      used_feature_max: feature max used for scaling (shape: k,)
    """
    texts_n = [normalize_text(t) for t in texts]
    X_text = vectorizer.transform(texts_n)

    # base numeric indicators (e.g., len_chars, len_tokens, digit_count, ttr, generic_hits)
    num = extract_indicators(texts_n, indicator_cfg)

    # similarity feature (nn similarity)
    nn = nearest_neighbor_similarity(
        texts=texts_n,
        reference_texts=similarity_reference_texts,
        vectorizer=similarity_vectorizer,
    ).reshape(-1, 1)

    # combine numeric indicators + similarity
    num_all = np.concatenate([num, nn], axis=1)  # expected: 6 numeric features

    # robust scaling: ensure feature_max matches actual feature count
    if num_feature_max is None or len(num_feature_max) != num_all.shape[1]:
        used_feature_max = np.maximum(num_all.max(axis=0), 1.0)
    else:
        used_feature_max = np.maximum(num_feature_max, 1.0)

    num_scaled = _scale_nonneg(num_all, used_feature_max)

    X_num = sparse.csr_matrix(num_scaled)
    X = sparse.hstack([X_text, X_num], format="csr")
    return X, num_all, nn.reshape(-1), used_feature_max


def cross_validate(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Stratified K-fold CV with leakage-safe similarity.

    For each fold:
      - fit TF-IDF on train texts
      - fit similarity vectorizer on train texts
      - compute numeric feature max on TRAIN only
      - build train/val features using same scaling
      - train NB, evaluate on val
    """
    texts = df[text_col].astype(str).tolist()
    y = df[label_col].astype(int).values

    # Robust CV for small datasets
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        raise ValueError(
            "Need at least 2 classes for cross-validation. "
            "Your dataset seems to contain only one label."
        )
    min_count = int(counts.min())
    if n_splits > min_count:
        new_splits = max(2, min_count)
        print(
            f"[warn] n_splits={n_splits} too large for the smallest class (min={min_count}). "
            f"Auto-using n_splits={new_splits}."
        )
        n_splits = new_splits

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    metrics = {"precision": [], "recall": [], "f1": [], "accuracy": [], "fpr": []}
    cfg = IndicatorConfig.default()

    for train_idx, val_idx in skf.split(texts, y):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Text TF-IDF vectorizer (fit on TRAIN only)
        vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)
        vec.fit([normalize_text(t) for t in train_texts])

        # Similarity vectorizer (fit on TRAIN reference only)
        sim_vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)
        sim_vec.fit([normalize_text(t) for t in train_texts])

        # Build TRAIN once to get feature_max based on TRAIN only
        X_train_tmp, num_train_raw, _, train_feature_max = build_features(
            texts=train_texts,
            vectorizer=vec,
            indicator_cfg=cfg,
            num_feature_max=None,
            similarity_reference_texts=None,
            similarity_vectorizer=sim_vec,
        )

        # Rebuild TRAIN with fixed scaling (optional but keeps logic explicit)
        X_train, _, _, _ = build_features(
            texts=train_texts,
            vectorizer=vec,
            indicator_cfg=cfg,
            num_feature_max=train_feature_max,
            similarity_reference_texts=None,
            similarity_vectorizer=sim_vec,
        )

        # Build VAL using scaling computed from TRAIN, similarity against TRAIN reference
        X_val, _, _, _ = build_features(
            texts=val_texts,
            vectorizer=vec,
            indicator_cfg=cfg,
            num_feature_max=train_feature_max,
            similarity_reference_texts=train_texts,
            similarity_vectorizer=sim_vec,
        )

        clf = MultinomialNB()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, average="binary", zero_division=0
        )
        acc = accuracy_score(y_val, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        metrics["precision"].append(float(prec))
        metrics["recall"].append(float(rec))
        metrics["f1"].append(float(f1))
        metrics["accuracy"].append(float(acc))
        metrics["fpr"].append(float(fpr))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def train_full(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> SavedModel:
    texts = df[text_col].astype(str).tolist()
    y = df[label_col].astype(int).values

    cfg = IndicatorConfig.default()

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)
    vec.fit([normalize_text(t) for t in texts])

    sim_vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)
    sim_vec.fit([normalize_text(t) for t in texts])

    # compute numeric max for scaling on FULL data (for final model)
    num = extract_indicators([normalize_text(t) for t in texts], cfg)
    nn = nearest_neighbor_similarity(texts, reference_texts=None, vectorizer=sim_vec).reshape(-1, 1)
    num_all = np.concatenate([num, nn], axis=1)
    num_feature_max = np.maximum(num_all.max(axis=0), 1.0)

    X, _, _, _ = build_features(
        texts=texts,
        vectorizer=vec,
        indicator_cfg=cfg,
        num_feature_max=num_feature_max,
        similarity_reference_texts=None,
        similarity_vectorizer=sim_vec,
    )

    clf = MultinomialNB()
    clf.fit(X, y)

    return SavedModel(vectorizer=vec, clf=clf, indicator_cfg=cfg, num_feature_max=num_feature_max)


def save_model(model: SavedModel, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    meta = {
        "text_vectorizer": "TfidfVectorizer(1-2gram, max_features=50000)",
        "classifier": "MultinomialNB",
        "num_features": ["len_chars", "len_tokens", "digit_count", "ttr", "generic_hits", "nn_similarity"],
        "score_formula": "credibility_score = round(100*(1-p_fake))",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_model(model_dir: str) -> SavedModel:
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        return pickle.load(f)


def score_reviews(
    model: SavedModel,
    texts: List[str],
    reference_texts: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Return dataframe with p_fake and credibility_score.

    If reference_texts is provided, nn similarity computed vs that corpus.
    If not, similarity computed within batch (MVP).
    """
    ref = reference_texts if reference_texts is not None else texts
    sim_vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=1)
    sim_vec.fit([normalize_text(t) for t in ref])

    X, _, _, _ = build_features(
        texts=texts,
        vectorizer=model.vectorizer,
        indicator_cfg=model.indicator_cfg,
        num_feature_max=model.num_feature_max,
        similarity_reference_texts=reference_texts,
        similarity_vectorizer=sim_vec,
    )

    proba = model.clf.predict_proba(X)
    p_fake = proba[:, 1]  # assume class 1 = fake
    score = np.rint(100.0 * (1.0 - p_fake)).astype(int)

    return pd.DataFrame({"p_fake": p_fake, "credibility_score": score})
