"""Classifying gene roles from amino acid sequences using a 1D Convolutional
Neural Network, trained via compiled Numba Adam backpropagation with exact max-pooling gradient routing.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
from numba import njit, prange


# ----------------------------------------------------------------------------------------------------------------------
# Scikit-learn Compatible Base Estimator & Classifier Mixin
# ----------------------------------------------------------------------------------------------------------------------
class BaseEstimator:
    """Zero-dependency drop-in replacement for sklearn.base.BaseEstimator."""

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Inspects __init__ parameters dynamically."""
        init_sig = inspect.signature(self.__init__)
        return {
            param.name: getattr(self, param.name)
            for param in init_sig.parameters.values()
            if param.name != "self" and param.kind != param.VAR_KEYWORD
        }

    def set_params(self, **params: object) -> Self:
        """Sets constructor parameters for GridSearch and Pipeline compatibility."""
        valid_params = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter '{key}' for {self.__class__.__name__}")
            setattr(self, key, value)
        return self


class ClassifierMixin:
    """Zero-dependency drop-in replacement for sklearn.base.ClassifierMixin."""

    _estimator_type = "classifier"

    def score(
        self,
        X: Sequence[np.ndarray] | tuple[np.ndarray, np.ndarray],
        y: Sequence[int] | np.ndarray,
    ) -> float:
        """Default mean accuracy score for cross-validation."""
        return float(np.mean(self.predict(X) == y))


# ----------------------------------------------------------------------------------------------------------------------
# GeneRoleClassifier Class
# ----------------------------------------------------------------------------------------------------------------------
class GeneRoleClassifier(BaseEstimator, ClassifierMixin):
    r"""Classifies `GeneRoles` to sequence tokens using a Multi-Scale 1D-CNN with Dual Pooling and Sequence Length.

    Learns representations from variable-length sequence token arrays using an embedding layer,
    multi-scale 1D convolutions (e.g. receptive fields 5, 9, 13), dual pooling (global max pooling
    + global average pooling) concatenated with normalized sequence length, and a dense
    classification layer. Optimization is performed via a compiled Numba Adam training kernel
    with exact gradient routing.

    Attributes:
        vocab_size: Number of discrete tokens in sequence vocabulary (default 23: 22 AMINO tokens + 1 unknown/pad).
        embed_dim: Dimensionality of token embedding vectors.
        num_filters: Number of 1D convolutional feature channels across all kernel sizes.
        kernel_size: Width of 1D convolution receptive field window (int), or tuple of multi-scale
            kernel sizes (e.g. `(5, 9, 13)`).
        learning_rate: Adam step size hyperparameter.
        epochs: Number of complete training epochs over dataset.
        batch_size: Mini-batch size for Adam gradient accumulation and updates.
        weight_decay: L2 regularization coefficient applied to model parameters.
        random_state: Optional seed for reproducible parameter initialization and sample shuffling.
    """

    __slots__ = (
        # Hyperparameters
        "vocab_size",
        "embed_dim",
        "num_filters",
        "kernel_size",
        "learning_rate",
        "epochs",
        "batch_size",
        "weight_decay",
        "random_state",
        "class_weight",
        # Fitted model attributes
        "classes_",
        "embedding_",
        "conv_w_",
        "conv_b_",
        "dense_w_",
        "dense_b_",
        "loss_history_",
        "lut_",
        "k_sizes_",
        "unique_ks_",
        "k_indices_",
    )

    def __init__(
        self,
        vocab_size: int = 23,
        embed_dim: int = 16,
        num_filters: int = 64,
        kernel_size: int | tuple[int, ...] = (5, 9, 13),
        learning_rate: float = 0.01,
        epochs: int = 20,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        random_state: int | None = None,
        class_weight: str | None = "balanced",
    ) -> None:
        """Initializes the GeneRoleClassifier with specified architecture hyperparameters."""
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.class_weight = class_weight

        self.classes_ = None
        self.embedding_ = None
        self.conv_w_ = None
        self.conv_b_ = None
        self.dense_w_ = None
        self.dense_b_ = None
        self.loss_history_ = None
        self.lut_ = None
        self.k_sizes_ = None
        self.unique_ks_ = None
        self.k_indices_ = None

    def _pack_ragged(
        self,
        X: Sequence[np.ndarray] | tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Packs sequence inputs into flat 1D token array and int64 offsets."""
        if (
            isinstance(X, tuple)
            and len(X) == 2
            and isinstance(X[0], np.ndarray)
            and isinstance(X[1], np.ndarray)
        ):
            flat = X[0].astype(np.int32, copy=False)
            offsets = X[1].astype(np.int64, copy=False)
            if len(offsets) > 0 and offsets[-1] != len(flat):
                offsets = np.append(offsets, len(flat))
            return flat, offsets

        offsets = np.zeros(len(X) + 1, dtype=np.int64)
        total_len = sum(len(seq) for seq in X)
        flat = np.empty(total_len, dtype=np.int32)
        curr = 0
        for i, seq in enumerate(X):
            n = len(seq)
            flat[curr : curr + n] = seq
            curr += n
            offsets[i + 1] = curr
        return flat, offsets

    def fit(
        self,
        X: Sequence[np.ndarray] | tuple[np.ndarray, np.ndarray],
        y: Sequence[int] | np.ndarray,
    ) -> Self:
        """Fits the 1D-CNN using Adam optimization over ragged token sequences.

        Args:
            X: Sequence of 1D integer token arrays (`list[np.ndarray]`), or `(flat_tokens, offsets)` tuple.
            y: 1D array-like of integer `GeneRoles` indices.

        Returns:
            Self: Fitted classifier instance.
        """
        y_arr = np.asarray(y, dtype=np.int8)
        if len(y_arr) == 0:
            raise ValueError("Training dataset cannot be empty.")

        self.classes_ = np.unique(y_arr).astype(np.int8)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(f"GeneRoleClassifier requires at least 2 distinct classes, found {n_classes}.")

        y_indexed = np.searchsorted(self.classes_, y_arr).astype(np.int32)

        if self.class_weight == "balanced":
            counts = np.bincount(y_indexed, minlength=n_classes).astype(np.float32)
            counts_safe = np.maximum(counts, 1.0)
            weights = float(len(y_indexed)) / (float(n_classes) * counts_safe)
            class_weights = (weights / np.mean(weights)).astype(np.float32)
        else:
            class_weights = np.ones(n_classes, dtype=np.float32)

        # Multi-scale kernel allocation and He-normal initialization
        if isinstance(self.kernel_size, int):
            k_tuple = (self.kernel_size,)
        else:
            k_tuple = tuple(int(k) for k in self.kernel_size)

        n_k = len(k_tuple)
        self.k_sizes_ = np.empty(self.num_filters, dtype=np.int32)
        base_count = self.num_filters // n_k
        remainder = self.num_filters % n_k
        idx = 0
        for i, k in enumerate(k_tuple):
            count = base_count + (1 if i < remainder else 0)
            self.k_sizes_[idx : idx + count] = k
            idx += count

        self.unique_ks_ = np.unique(self.k_sizes_).astype(np.int32)
        self.k_indices_ = np.array(
            [np.where(self.unique_ks_ == k)[0][0] for k in self.k_sizes_],
            dtype=np.int32,
        )
        max_k = int(np.max(self.unique_ks_))

        rng = np.random.default_rng(self.random_state)
        self.embedding_ = rng.normal(0, 0.1, (self.vocab_size, self.embed_dim)).astype(np.float32)
        self.conv_w_ = np.zeros((self.num_filters, self.embed_dim, max_k), dtype=np.float32)
        for oc in range(self.num_filters):
            K = int(self.k_sizes_[oc])
            std = float(np.sqrt(2.0 / (self.embed_dim * K)))
            self.conv_w_[oc, :, :K] = rng.normal(0.0, std, (self.embed_dim, K)).astype(np.float32)
        self.conv_b_ = np.zeros(self.num_filters, dtype=np.float32)
        feat_dim = 2 * self.num_filters + 1
        self.dense_w_ = rng.normal(
            0,
            np.sqrt(2 / feat_dim),
            (feat_dim, n_classes),
        ).astype(np.float32)
        self.dense_b_ = np.zeros(n_classes, dtype=np.float32)

        X_flat, offsets = self._pack_ragged(X)
        if len(offsets) - 1 != len(y_arr):
            raise ValueError(
                f"Number of samples in X ({len(offsets) - 1}) does not match y ({len(y_arr)})."
            )

        seed = -1 if self.random_state is None else int(self.random_state)

        # Train via compiled Numba Adam backprop kernel
        (
            self.embedding_,
            self.conv_w_,
            self.conv_b_,
            self.dense_w_,
            self.dense_b_,
            loss_hist,
        ) = _train_adam(
            X_flat,
            offsets,
            y_indexed,
            class_weights,
            self.embedding_,
            self.conv_w_,
            self.conv_b_,
            self.dense_w_,
            self.dense_b_,
            self.k_sizes_,
            self.unique_ks_,
            self.k_indices_,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.weight_decay,
            seed,
        )
        self.loss_history_ = list(loss_hist)
        self.lut_ = _compute_lut(self.embedding_, self.conv_w_, self.k_sizes_)
        return self

    def predict_proba(
        self,
        X: Sequence[np.ndarray] | tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Predicts class probabilities for each sequence in X.

        Args:
            X: Sequence of 1D integer token arrays, or `(flat_tokens, offsets)` tuple.

        Returns:
            np.ndarray: Float32 array of shape `(n_samples, n_classes)` with softmax probabilities.
        """
        self._check_is_fitted()
        X_flat, offsets = self._pack_ragged(X)
        return _forward_batch(
            X_flat,
            offsets,
            self.lut_,  # type: ignore[arg-type]
            self.conv_b_,  # type: ignore[arg-type]
            self.dense_w_,  # type: ignore[arg-type]
            self.dense_b_,  # type: ignore[arg-type]
            self.k_sizes_,  # type: ignore[arg-type]
        )

    def predict(
        self,
        X: Sequence[np.ndarray[tuple[int], np.dtype[np.uint8]]] | tuple[np.ndarray[tuple[int], np.dtype[np.uint8]], np.ndarray[tuple[int], np.dtype[np.uint32]]],
        min_confidence: float = 0.0,
    ) -> np.ndarray[tuple[np.ndarray, np.dtype[np.uint8]]]:
        """Predicts most likely GeneRole integer index for each sequence in X.

        Args:
            X: Sequence of 1D integer token arrays, or `(flat_tokens, offsets)` tuple.
            min_confidence: Minimum predicted probability required. If max predicted
                probability is below this threshold, assigns to UNKNOWN (role 29).

        Returns:
            np.ndarray: Int8 array of shape `(n_samples,)` of predicted role indices.
        """
        probs = self.predict_proba(X)
        top_indices = np.argmax(probs, axis=1)
        top_classes = self.classes_[top_indices].astype(np.int8)  # type: ignore[index]
        if min_confidence > 0.0:
            max_probs = np.max(probs, axis=1)
            return np.where(max_probs >= min_confidence, top_classes, np.int8(29)).astype(np.int8)
        return top_classes

    def _check_is_fitted(self) -> None:
        """Verifies that all required model parameters are fitted."""
        for attr in ("classes_", "embedding_", "conv_w_", "conv_b_", "dense_w_", "dense_b_", "k_sizes_"):
            if not hasattr(self, attr) or getattr(self, attr) is None:
                raise RuntimeError(
                    f"This {self.__class__.__name__} instance is not fitted yet. "
                    "Call 'fit' or load pre-trained weights first."
                )
        if getattr(self, "lut_", None) is None:
            self.lut_ = _compute_lut(self.embedding_, self.conv_w_, self.k_sizes_)  # type: ignore[arg-type]

    def save(self, path: Path | str) -> None:
        """Saves model weights, classes, and hyperparameters to a compressed .npz archive.

        Args:
            path: Destination file path (with or without `.npz` extension).
        """
        self._check_is_fitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        cw_flag = 1 if self.class_weight == "balanced" else 0
        first_k = int(self.k_sizes_[0]) if hasattr(self, "k_sizes_") and self.k_sizes_ is not None else 7
        meta = np.array([
            self.vocab_size,
            self.embed_dim,
            self.num_filters,
            first_k,
            self.batch_size,
            self.epochs,
            cw_flag,
        ], dtype=np.int64)

        float_meta = np.array([
            self.learning_rate,
            self.weight_decay,
            -1.0 if self.random_state is None else float(self.random_state),
        ], dtype=np.float64)

        np.savez_compressed(
            p,
            meta=meta,
            float_meta=float_meta,
            classes=self.classes_,
            embedding=self.embedding_,
            conv_w=self.conv_w_,
            conv_b=self.conv_b_,
            dense_w=self.dense_w_,
            dense_b=self.dense_b_,
            k_sizes=self.k_sizes_,
            loss_history=np.array(self.loss_history_ or [], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """Loads model weights, classes, and hyperparameters from a compressed .npz archive.

        Args:
            path: Source file path.

        Returns:
            GeneRoleClassifier: Restored model instance with fitted weights.
        """
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")

        with np.load(p) as data:
            meta = data["meta"]
            float_meta = data["float_meta"]
            rs = None if float_meta[2] < 0 else int(float_meta[2])
            cw = "balanced" if (len(meta) > 6 and meta[6] == 1) else None

            num_filters = int(meta[2])
            if "k_sizes" in data:
                k_sizes = data["k_sizes"].astype(np.int32)
                unique_ks = np.unique(k_sizes)
                if len(unique_ks) == 1:
                    ks_param: int | tuple[int, ...] = int(unique_ks[0])
                else:
                    ks_param = tuple(int(x) for x in unique_ks)
            else:
                ks_param = int(meta[3])
                k_sizes = np.full(num_filters, ks_param, dtype=np.int32)

            inst = cls(
                vocab_size=int(meta[0]),
                embed_dim=int(meta[1]),
                num_filters=num_filters,
                kernel_size=ks_param,
                batch_size=int(meta[4]),
                epochs=int(meta[5]),
                learning_rate=float(float_meta[0]),
                weight_decay=float(float_meta[1]),
                random_state=rs,
                class_weight=cw,
            )
            inst.classes_ = data["classes"].astype(np.int8)
            inst.embedding_ = data["embedding"].astype(np.float32)
            inst.conv_w_ = data["conv_w"].astype(np.float32)
            inst.conv_b_ = data["conv_b"].astype(np.float32)
            inst.dense_w_ = data["dense_w"].astype(np.float32)
            inst.dense_b_ = data["dense_b"].astype(np.float32)
            inst.k_sizes_ = k_sizes
            inst.unique_ks_ = np.unique(k_sizes).astype(np.int32)
            inst.k_indices_ = np.array(
                [np.where(inst.unique_ks_ == k)[0][0] for k in k_sizes],
                dtype=np.int32,
            )
            inst.loss_history_ = list(data["loss_history"].tolist())
            inst.lut_ = _compute_lut(inst.embedding_, inst.conv_w_, inst.k_sizes_)
            return inst

    def __repr__(self) -> str:
        """Returns informative string representation of estimator."""
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        fitted_str = (
            f" [fitted with {len(self.classes_)} classes]"
            if hasattr(self, "classes_") and self.classes_ is not None
            else ""
        )
        return f"{self.__class__.__name__}({param_str}){fitted_str}"


# ----------------------------------------------------------------------------------------------------------------------
# High-Performance Numba Kernels
# ----------------------------------------------------------------------------------------------------------------------
@njit(fastmath=True)
def _compute_lut(embedding: np.ndarray, conv_w: np.ndarray, k_sizes: np.ndarray) -> np.ndarray:
    """Precomputes token-position-channel lookup table: (vocab_size, max_k, out_channels)."""
    vocab_size, embed_dim = embedding.shape
    out_channels, in_channels, max_k = conv_w.shape
    lut = np.zeros((vocab_size, max_k, out_channels), dtype=np.float32)
    for oc in range(out_channels):
        K = k_sizes[oc]
        for k in range(K):
            for t in range(vocab_size):
                s = 0.0
                for ic in range(in_channels):
                    s += embedding[t, ic] * conv_w[oc, ic, k]
                lut[t, k, oc] = s
    return lut


@njit(parallel=True, fastmath=True)
def _forward_batch(
    sequences_flat: np.ndarray,
    offsets: np.ndarray,
    lut: np.ndarray,  # (vocab_size, max_k, out_channels)
    conv_b: np.ndarray,  # (out_channels,)
    dense_w: np.ndarray,  # (2 * out_channels + 1, n_classes)
    dense_b: np.ndarray,  # (n_classes,)
    k_sizes: np.ndarray,  # (out_channels,)
) -> np.ndarray:
    """Parallelized forward inference kernel using multi-scale token LUT, dual pooling, length, and softmax."""
    n_samples = len(offsets) - 1
    out_channels = len(conv_b)
    feat_dim = 2 * out_channels + 1
    n_classes = len(dense_b)
    probs = np.empty((n_samples, n_classes), dtype=np.float32)

    for i in prange(n_samples):
        start = offsets[i]
        end = offsets[i + 1]
        seq_len = end - start

        h = np.zeros(feat_dim, dtype=np.float32)
        h[2 * out_channels] = np.log(max(float(seq_len), 1.0)) / 7.0

        for oc in range(out_channels):
            K = k_sizes[oc]
            if seq_len >= K:
                out_len = seq_len - K + 1
                inv_out_len = 1.0 / float(out_len)
                max_val = 0.0
                sum_val = 0.0
                for pos in range(out_len):
                    val = conv_b[oc]
                    for k in range(K):
                        tok = sequences_flat[start + pos + k]
                        val += lut[tok, k, oc]
                    sum_val += val
                    if val > max_val:
                        max_val = val
                h[oc] = max_val
                h[out_channels + oc] = sum_val * inv_out_len

        # Dense classification layer
        max_logit = -1e9
        logits = np.empty(n_classes, dtype=np.float32)
        for c in range(n_classes):
            s = dense_b[c]
            for j in range(feat_dim):
                s += h[j] * dense_w[j, c]
            logits[c] = s
            if s > max_logit:
                max_logit = s

        # Numerically stable Softmax
        exp_sum = 0.0
        for c in range(n_classes):
            e = np.exp(logits[c] - max_logit)
            logits[c] = e
            exp_sum += e

        for c in range(n_classes):
            probs[i, c] = logits[c] / exp_sum

    return probs


@njit(fastmath=True)
def _train_adam(
    sequences_flat: np.ndarray,
    offsets: np.ndarray,
    y_indexed: np.ndarray,
    class_weights: np.ndarray,
    embedding: np.ndarray,
    conv_w: np.ndarray,
    conv_b: np.ndarray,
    dense_w: np.ndarray,
    dense_b: np.ndarray,
    k_sizes: np.ndarray,
    unique_ks: np.ndarray,
    k_indices: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mini-batch Adam training kernel with multi-scale kernels, dual pooling, length, and backprop."""
    if seed >= 0:
        np.random.seed(seed)

    n_samples = len(offsets) - 1
    out_channels, in_channels, max_k = conv_w.shape
    feat_dim = 2 * out_channels + 1
    n_classes = len(dense_b)
    vocab_size, embed_dim = embedding.shape
    num_unique = len(unique_ks)

    # Adam moment buffers
    m_emb = np.zeros_like(embedding)
    v_emb = np.zeros_like(embedding)
    m_cw = np.zeros_like(conv_w)
    v_cw = np.zeros_like(conv_w)
    m_cb = np.zeros_like(conv_b)
    v_cb = np.zeros_like(conv_b)
    m_dw = np.zeros_like(dense_w)
    v_dw = np.zeros_like(dense_w)
    m_db = np.zeros_like(dense_b)
    v_db = np.zeros_like(dense_b)

    # Mini-batch gradient accumulators
    grad_emb = np.zeros_like(embedding)
    grad_cw = np.zeros_like(conv_w)
    grad_cb = np.zeros_like(conv_b)
    grad_dw = np.zeros_like(dense_w)
    grad_db = np.zeros_like(dense_b)

    loss_history = np.empty(epochs, dtype=np.float32)
    step = 0
    lut = _compute_lut(embedding, conv_w, k_sizes)

    # Preallocated per-sample buffers
    h = np.zeros(feat_dim, dtype=np.float32)
    d_h = np.zeros(feat_dim, dtype=np.float32)
    argmax_pos = np.full(out_channels, -1, dtype=np.int32)
    logits = np.empty(n_classes, dtype=np.float32)
    probs = np.empty(n_classes, dtype=np.float32)
    E_mean = np.zeros((num_unique, max_k, in_channels), dtype=np.float32)
    V = np.zeros((num_unique, in_channels, max_k), dtype=np.float32)

    indices = np.arange(n_samples)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        np.random.shuffle(indices)

        for s_idx in range(n_samples):
            i = indices[s_idx]
            start = offsets[i]
            end = offsets[i + 1]
            seq_len = end - start

            for j in range(feat_dim):
                h[j] = 0.0
            h[2 * out_channels] = np.log(max(float(seq_len), 1.0)) / 7.0

            for oc in range(out_channels):
                argmax_pos[oc] = -1
                K = k_sizes[oc]
                if seq_len >= K:
                    out_len = seq_len - K + 1
                    inv_out_len = 1.0 / float(out_len)
                    max_val = 0.0
                    sum_val = 0.0
                    for pos in range(out_len):
                        val = conv_b[oc]
                        for k in range(K):
                            tok = sequences_flat[start + pos + k]
                            val += lut[tok, k, oc]
                        sum_val += val
                        if val > max_val:
                            max_val = val
                            argmax_pos[oc] = pos
                    h[oc] = max_val
                    h[out_channels + oc] = sum_val * inv_out_len

            max_logit = -1e9
            for c in range(n_classes):
                s = dense_b[c]
                for j in range(feat_dim):
                    s += h[j] * dense_w[j, c]
                logits[c] = s
                if s > max_logit:
                    max_logit = s

            exp_sum = 0.0
            for c in range(n_classes):
                e = np.exp(logits[c] - max_logit)
                probs[c] = e
                exp_sum += e

            for c in range(n_classes):
                probs[c] = probs[c] / exp_sum

            target = y_indexed[i]
            w = class_weights[target]
            epoch_loss += -w * np.log(probs[target] + 1e-10)

            # Gradient w.r.t. dense layer
            for c in range(n_classes):
                d_logit = w * (probs[c] - (1.0 if c == target else 0.0))
                grad_db[c] += d_logit
                for j in range(feat_dim):
                    grad_dw[j, c] += h[j] * d_logit

            # Backprop into feature vector h
            for j in range(feat_dim):
                s = 0.0
                for c in range(n_classes):
                    d_logit = w * (probs[c] - (1.0 if c == target else 0.0))
                    s += dense_w[j, c] * d_logit
                d_h[j] = s

            # 1. Backprop avg_pool (analytically factored over unique kernel sizes)
            for u in range(num_unique):
                for k in range(max_k):
                    for ic in range(in_channels):
                        E_mean[u, k, ic] = 0.0
                        V[u, ic, k] = 0.0

            for oc in range(out_channels):
                K = k_sizes[oc]
                if seq_len >= K:
                    d_avg_oc = d_h[out_channels + oc]
                    grad_cb[oc] += d_avg_oc
                    u = k_indices[oc]
                    for ic in range(in_channels):
                        for k in range(K):
                            V[u, ic, k] += d_avg_oc * conv_w[oc, ic, k]

            for u in range(num_unique):
                K = unique_ks[u]
                if seq_len >= K:
                    out_len = seq_len - K + 1
                    inv_out_len = 1.0 / float(out_len)
                    for k in range(K):
                        for pos in range(out_len):
                            tok = sequences_flat[start + pos + k]
                            for ic in range(in_channels):
                                E_mean[u, k, ic] += embedding[tok, ic]
                                grad_emb[tok, ic] += inv_out_len * V[u, ic, k]

                    for k in range(K):
                        for ic in range(in_channels):
                            E_mean[u, k, ic] *= inv_out_len

            for oc in range(out_channels):
                K = k_sizes[oc]
                if seq_len >= K:
                    d_avg_oc = d_h[out_channels + oc]
                    u = k_indices[oc]
                    for ic in range(in_channels):
                        for k in range(K):
                            grad_cw[oc, ic, k] += d_avg_oc * E_mean[u, k, ic]

            # 2. Backprop max_pool
            for oc in range(out_channels):
                pos = argmax_pos[oc]
                K = k_sizes[oc]
                if pos >= 0 and h[oc] > 0.0:
                    d_max_oc = d_h[oc]
                    grad_cb[oc] += d_max_oc
                    for ic in range(in_channels):
                        for k in range(K):
                            tok = sequences_flat[start + pos + k]
                            grad_cw[oc, ic, k] += d_max_oc * embedding[tok, ic]
                            grad_emb[tok, ic] += d_max_oc * conv_w[oc, ic, k]

            batch_count += 1
            if batch_count == batch_size or s_idx == n_samples - 1:
                step += 1
                inv_bc = 1.0 / batch_count
                beta1 = 0.9
                beta2 = 0.999
                eps = 1e-8
                bc1 = 1.0 - beta1 ** step
                bc2 = 1.0 - beta2 ** step
                step_size = learning_rate * np.sqrt(bc2) / bc1

                # Update dense_b
                for c in range(n_classes):
                    g = grad_db[c] * inv_bc
                    m_db[c] = beta1 * m_db[c] + (1.0 - beta1) * g
                    v_db[c] = beta2 * v_db[c] + (1.0 - beta2) * g * g
                    dense_b[c] -= step_size * m_db[c] / (np.sqrt(v_db[c]) + eps)
                    grad_db[c] = 0.0

                # Update dense_w
                for j in range(feat_dim):
                    for c in range(n_classes):
                        g = grad_dw[j, c] * inv_bc + weight_decay * dense_w[j, c]
                        m_dw[j, c] = beta1 * m_dw[j, c] + (1.0 - beta1) * g
                        v_dw[j, c] = beta2 * v_dw[j, c] + (1.0 - beta2) * g * g
                        dense_w[j, c] -= step_size * m_dw[j, c] / (np.sqrt(v_dw[j, c]) + eps)
                        grad_dw[j, c] = 0.0

                # Update conv_b
                for oc in range(out_channels):
                    g = grad_cb[oc] * inv_bc
                    m_cb[oc] = beta1 * m_cb[oc] + (1.0 - beta1) * g
                    v_cb[oc] = beta2 * v_cb[oc] + (1.0 - beta2) * g * g
                    conv_b[oc] -= step_size * m_cb[oc] / (np.sqrt(v_cb[oc]) + eps)
                    grad_cb[oc] = 0.0

                # Update conv_w
                for oc in range(out_channels):
                    K = k_sizes[oc]
                    for ic in range(in_channels):
                        for k in range(K):
                            g = grad_cw[oc, ic, k] * inv_bc + weight_decay * conv_w[oc, ic, k]
                            m_cw[oc, ic, k] = beta1 * m_cw[oc, ic, k] + (1.0 - beta1) * g
                            v_cw[oc, ic, k] = beta2 * v_cw[oc, ic, k] + (1.0 - beta2) * g * g
                            conv_w[oc, ic, k] -= step_size * m_cw[oc, ic, k] / (np.sqrt(v_cw[oc, ic, k]) + eps)
                            grad_cw[oc, ic, k] = 0.0

                # Update embedding
                for tok in range(vocab_size):
                    for ic in range(embed_dim):
                        g = grad_emb[tok, ic] * inv_bc + weight_decay * embedding[tok, ic]
                        m_emb[tok, ic] = beta1 * m_emb[tok, ic] + (1.0 - beta1) * g
                        v_emb[tok, ic] = beta2 * v_emb[tok, ic] + (1.0 - beta2) * g * g
                        embedding[tok, ic] -= step_size * m_emb[tok, ic] / (np.sqrt(v_emb[tok, ic]) + eps)
                        grad_emb[tok, ic] = 0.0

                lut = _compute_lut(embedding, conv_w, k_sizes)
                batch_count = 0

        loss_history[epoch] = epoch_loss / n_samples

    return embedding, conv_w, conv_b, dense_w, dense_b, loss_history