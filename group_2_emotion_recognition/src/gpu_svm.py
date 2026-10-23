"""Exact RBF-kernel SVM whose kernel matrix is computed with PyTorch on the GPU (CUDA).

Why: training an RBF-SVM on ~30k FER images with ~1,000 features is dominated by kernel
evaluations (each costs O(d)). Here the full Gram matrix K = exp(-gamma * ||x_i - x_j||^2) is
computed once on the GPU and handed to scikit-learn's SVC(kernel="precomputed"), i.e. the very
same libsvm solver. The model is mathematically the same as SVC(kernel="rbf") (up to float32
rounding of the distances); only the expensive part runs on CUDA.

* Works without a GPU (PyTorch on CPU) and without PyTorch (NumPy fallback), so the saved models
  can be used by demo_emotion.py on any machine.
* The Gram matrix needs n^2 * 8 bytes of RAM (33k samples -> 8.7 GB). Above `max_kernel_n`
  samples the estimator falls back to the classic SVC(kernel="rbf").

This class lives in src/ (not in a notebook) only so that pickled models can be loaded by the demo.
"""
import os

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

MAX_KERNEL_N = int(os.environ.get("MAX_KERNEL_N", 33000))


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def device():
    t = _torch()
    if t is None:
        return "numpy"
    return "cuda" if t.cuda.is_available() else "cpu"


def rbf_kernel(A, B, gamma, block=2048):
    """exp(-gamma * ||a - b||^2) for all rows of A and B, as a float64 NumPy array."""
    A = np.asarray(A, np.float32)
    B = np.asarray(B, np.float32)
    out = np.empty((len(A), len(B)), np.float64)
    t = _torch()
    if t is None:
        bn = (B.astype(np.float64) ** 2).sum(1)
        for i in range(0, len(A), block):
            a = A[i:i + block].astype(np.float64)
            d = (a ** 2).sum(1)[:, None] + bn[None, :] - 2 * a @ B.T.astype(np.float64)
            out[i:i + block] = np.exp(-gamma * np.maximum(d, 0))
        return out
    dev = device()
    with t.no_grad():
        Bt = t.as_tensor(B, device=dev)
        bn = (Bt * Bt).sum(1)
        for i in range(0, len(A), block):
            At = t.as_tensor(A[i:i + block], device=dev)
            d = (At * At).sum(1)[:, None] + bn[None, :] - 2 * (At @ Bt.T)
            out[i:i + block] = t.exp(-gamma * d.clamp_min_(0)).double().cpu().numpy()
    return out


class GPUKernelSVC(ClassifierMixin, BaseEstimator):
    """Drop-in replacement for SVC(kernel="rbf") with the kernel computed on the GPU.

    gamma_mult multiplies gamma (useful to tune gamma relative to the data-dependent "scale" value).
    """

    def __init__(self, C=1.0, gamma="scale", gamma_mult=1.0, class_weight=None, max_kernel_n=MAX_KERNEL_N,
                 cache_size=2000, random_state=None):
        self.C = C
        self.gamma = gamma
        self.gamma_mult = gamma_mult
        self.class_weight = class_weight
        self.max_kernel_n = max_kernel_n
        self.cache_size = cache_size
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, np.float32)
        # same definition as scikit-learn's gamma="scale"
        base = 1.0 / (X.shape[1] * X.var()) if self.gamma == "scale" else float(self.gamma)
        self.gamma_ = base * self.gamma_mult
        self.precomputed_ = len(X) <= self.max_kernel_n
        if self.precomputed_:
            K = rbf_kernel(X, X, self.gamma_)
            self.svc_ = SVC(C=self.C, kernel="precomputed", class_weight=self.class_weight,
                            cache_size=self.cache_size, random_state=self.random_state)
            self.svc_.fit(K, y, sample_weight=sample_weight)
            del K
            self.support_vectors_ = X[self.svc_.support_]
            self.n_train_ = len(X)
        else:
            self.svc_ = SVC(C=self.C, kernel="rbf", gamma=self.gamma_, class_weight=self.class_weight,
                            cache_size=self.cache_size, random_state=self.random_state)
            self.svc_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.svc_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def _apply(self, method, X, block=1024):
        X = np.asarray(X, np.float32)
        if not self.precomputed_:
            return getattr(self.svc_, method)(X)
        out = []
        for i in range(0, len(X), block):
            Ksv = rbf_kernel(X[i:i + block], self.support_vectors_, self.gamma_)
            K = np.zeros((len(Ksv), self.n_train_))   # libsvm only reads the support-vector columns
            K[:, self.svc_.support_] = Ksv
            out.append(getattr(self.svc_, method)(K))
        return np.concatenate(out)

    def predict(self, X):
        return self._apply("predict", X)

    def decision_function(self, X):
        return self._apply("decision_function", X)
