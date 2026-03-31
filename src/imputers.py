from __future__ import annotations

from typing import Dict, Hashable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class LastNMeanImputer(BaseEstimator, TransformerMixin):
    """Impute NaNs per class using mean of last N observed rows per feature."""

    def __init__(self, n_last: int = 10):
        self.n_last = n_last

    def fit(self, X, y):
        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        if x_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if len(y_arr) != len(x_arr):
            raise ValueError("X and y must have same number of rows.")

        self.n_features_in_ = x_arr.shape[1]
        self.classes_ = np.unique(y_arr)
        self.global_means_ = np.nanmean(x_arr, axis=0)
        self.class_feature_means_: Dict[Hashable, np.ndarray] = {}

        for cls in self.classes_:
            cls_rows = x_arr[y_arr == cls]
            means = np.zeros(self.n_features_in_, dtype=float)
            for idx in range(self.n_features_in_):
                col = cls_rows[:, idx]
                observed = col[~np.isnan(col)]
                if observed.size == 0:
                    means[idx] = self.global_means_[idx]
                else:
                    means[idx] = observed[-self.n_last :].mean()
            self.class_feature_means_[cls] = means
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["class_feature_means_", "global_means_"])
        x_arr = np.asarray(X, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.shape[1] != self.n_features_in_:
            raise ValueError("Feature count mismatch during transform.")

        out = x_arr.copy()
        for row_idx in range(out.shape[0]):
            row = out[row_idx]
            missing_idx = np.where(np.isnan(row))[0]
            if missing_idx.size == 0:
                continue

            cls = None
            if y is not None:
                cls = np.asarray(y)[row_idx]
            fill_source = self.class_feature_means_.get(cls, self.global_means_)
            row[missing_idx] = fill_source[missing_idx]
            fallback_missing = np.isnan(row)
            row[fallback_missing] = self.global_means_[fallback_missing]
            out[row_idx] = row
        return out
