"""Effort model: text + ticket metadata -> hours of logged work.

Two learners are blended in log-hours space:
  * a Ridge regression over TF-IDF text and one-hot metadata, and
  * a nearest-neighbour vote over the most similar past tickets.
The blend weight and the uncertainty band are fitted on out-of-fold predictions,
so the reported range reflects how wrong the model actually tends to be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from .records import Ticket

MAX_DESCRIPTION_CHARS = 2000


def ticket_text(t: Ticket) -> str:
    parts = [t.summary, t.summary, t.parent_summary, " ".join(t.labels), " ".join(t.components),
             t.description[:MAX_DESCRIPTION_CHARS]]
    return " ".join(p for p in parts if p)


def ticket_categories(t: Ticket) -> list[str]:
    return [t.project or "?", t.issue_type or "?", t.priority or "?", f"{t.project}|{t.issue_type}",
            t.assignee or "?", f"{t.assignee}|{t.issue_type}"]


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, cum[-1] / 2.0)])


@dataclass
class Similar:
    key: str
    summary: str
    hours_spent: float
    similarity: float


@dataclass
class Estimate:
    hours: float
    low: float
    high: float
    similar: list[Similar]
    reference_hours: list[float]  # actual hours of the closest past tickets (the "outside view")
    team_estimate_bias: float | None  # median actual/estimate for this project+type; >1 means under-estimating
    bias_scope: str = ""              # which history the bias came from, e.g. "MT Bug"


class EffortModel:
    def __init__(self, n_neighbors: int = 7, ridge_alpha: float = 1.0, interval: tuple[float, float] = (0.1, 0.9)):
        self.n_neighbors = n_neighbors
        self.ridge_alpha = ridge_alpha
        self.interval = interval

    # ---- features -------------------------------------------------------------------------
    def _fit_features(self, tickets: Sequence[Ticket]) -> sparse.csr_matrix:
        texts = [ticket_text(t) for t in tickets]
        self.words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True,
                                     stop_words="english")
        self.chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=30000,
                                     sublinear_tf=True)
        self.cats = OneHotEncoder(handle_unknown="ignore")
        w = self.words.fit_transform(texts)
        c = self.chars.fit_transform(texts)
        k = self.cats.fit_transform([ticket_categories(t) for t in tickets])
        return sparse.hstack([w, 0.5 * c, k, self._numeric(tickets)]).tocsr()

    def _features(self, tickets: Sequence[Ticket]) -> sparse.csr_matrix:
        texts = [ticket_text(t) for t in tickets]
        w = self.words.transform(texts)
        c = self.chars.transform(texts)
        k = self.cats.transform([ticket_categories(t) for t in tickets])
        return sparse.hstack([w, 0.5 * c, k, self._numeric(tickets)]).tocsr()

    @staticmethod
    def _numeric(tickets: Sequence[Ticket]) -> sparse.csr_matrix:
        lengths = np.array([[np.log1p(len(t.description.split()))] for t in tickets]) / 5.0
        return sparse.csr_matrix(lengths)

    def _similarity_space(self, tickets: Sequence[Ticket]) -> sparse.csr_matrix:
        # Neighbour search uses text only: rows are L2-normalised so dot product = cosine.
        return self.words.transform([ticket_text(t) for t in tickets])

    # ---- learners -------------------------------------------------------------------------
    def _knn_predict(self, sims: np.ndarray, y: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        if exclude_self:
            np.fill_diagonal(sims, -1.0)
        k = min(self.n_neighbors, sims.shape[1])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out = np.empty(sims.shape[0])
        for i, row in enumerate(idx):
            w = np.clip(sims[i, row], 1e-3, None) ** 2
            out[i] = weighted_median(y[row], w)
        return out

    def fit(self, tickets: Sequence[Ticket]) -> "EffortModel":
        tickets = [t for t in tickets if t.hours_spent and t.hours_spent > 0]
        if len(tickets) < 30:
            raise ValueError(f"Need at least 30 tickets with logged time to train, got {len(tickets)}.")
        y = np.log1p([t.hours_spent for t in tickets])
        # Cap extreme outliers (e.g. a ticket used as a long-running time bucket).
        self.y_cap = float(np.quantile(y, 0.99))
        y = np.minimum(y, self.y_cap)

        # Out-of-fold predictions to choose the blend weight and calibrate the interval.
        oof_ridge, oof_knn = np.zeros_like(y), np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(tickets):
            sub = EffortModel(self.n_neighbors, self.ridge_alpha)
            Xtr = sub._fit_features([tickets[i] for i in tr])
            Xte = sub._features([tickets[i] for i in te])
            oof_ridge[te] = Ridge(alpha=self.ridge_alpha).fit(Xtr, y[tr]).predict(Xte)
            S = (sub._similarity_space([tickets[i] for i in te]) @ sub._similarity_space([tickets[i] for i in tr]).T)
            oof_knn[te] = sub._knn_predict(S.toarray(), y[tr])

        weights = np.linspace(0, 1, 11)
        errs = [np.median(np.abs(w * oof_ridge + (1 - w) * oof_knn - y)) for w in weights]
        self.blend = float(weights[int(np.argmin(errs))])
        resid = y - (self.blend * oof_ridge + (1 - self.blend) * oof_knn)
        self.resid_lo, self.resid_hi = (float(q) for q in np.quantile(resid, self.interval))
        self.cv_median_abs_log_error = float(np.min(errs))

        # Final fit on everything.
        X = self._fit_features(tickets)
        self.ridge = Ridge(alpha=self.ridge_alpha).fit(X, y)
        self.train_sim = self._similarity_space(tickets)
        self.train_y = y
        self.train_meta = [(t.key, t.summary, t.hours_spent) for t in tickets]
        self._fit_bias(tickets)
        return self

    def _fit_bias(self, tickets: Sequence[Ticket]) -> None:
        ratios: dict[tuple[str, str], list[float]] = {}
        for t in tickets:
            if t.hours_estimated and t.hours_estimated > 0 and t.hours_spent:
                for key in ((t.project, t.issue_type), (t.project, "*")):
                    ratios.setdefault(key, []).append(t.hours_spent / t.hours_estimated)
        self.bias = {k: float(np.median(v)) for k, v in ratios.items() if len(v) >= 5}

    # ---- inference ------------------------------------------------------------------------
    def predict_log(self, tickets: Sequence[Ticket]) -> tuple[np.ndarray, np.ndarray]:
        ridge = self.ridge.predict(self._features(tickets))
        sims = (self._similarity_space(tickets) @ self.train_sim.T).toarray()
        knn = self._knn_predict(sims, self.train_y)
        pred = np.minimum(self.blend * ridge + (1 - self.blend) * knn, self.y_cap)
        return pred, sims

    def estimate(self, ticket: Ticket, n_similar: int = 5, reference_size: int = 15) -> Estimate:
        pred, sims = self.predict_log([ticket])
        p, row = pred[0], sims[0]
        order = np.argsort(-row)
        top = order[:n_similar]
        similar = [Similar(self.train_meta[i][0], self.train_meta[i][1], self.train_meta[i][2], float(row[i]))
                   for i in top if row[i] > 0]
        if (ticket.project, ticket.issue_type) in self.bias:
            bias, scope = self.bias[(ticket.project, ticket.issue_type)], f"{ticket.project} {ticket.issue_type}"
        else:
            bias, scope = self.bias.get((ticket.project, "*")), ticket.project
        return Estimate(
            hours=float(np.expm1(p)),
            low=float(max(np.expm1(p + self.resid_lo), 0.0)),
            high=float(np.expm1(p + self.resid_hi)),
            similar=similar,
            reference_hours=[self.train_meta[i][2] for i in order[:reference_size]],
            team_estimate_bias=bias,
            bias_scope=scope,
        )
