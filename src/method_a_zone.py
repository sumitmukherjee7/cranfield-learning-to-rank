"""
Method A - Boolean weighted zone scoring with weights learned by grid search.

For a query q and document d, every zone z in {title, author, biblio, body}
gives a Boolean match score s_z(q, d).  The ranking score is

        score(q, d) = sum_z  g_z * s_z(q, d),      g_z >= 0,  sum_z g_z = 1

and the weights g are learned by exhaustive grid search over the simplex
using the training queries and their relevance judgements.

Four definitions of the per-zone Boolean match are supported:
    "fraction" - fraction of the query terms present in the zone
                 (each term contributes a 0/1 Boolean match)
    "any"      - 1 if at least one query term occurs in the zone (Boolean OR)
    "all"      - 1 if every query term occurs in the zone (Boolean AND)
    "idf"      - idf-weighted fraction: sum of idf(t) over the query terms
                 present in the zone / sum of idf(t) over all query terms
                 (still a Boolean match per term, but rare terms count more)
"""
import itertools

import numpy as np

from cranfield import FIELDS, average_precision, query_terms


def zone_features(index, qtext, mode="fraction"):
    """(N x 4) matrix of Boolean zone match scores for one query."""
    terms = query_terms(qtext)
    S = np.zeros((index.N, len(FIELDS)))
    if not terms:
        return S
    for j, f in enumerate(FIELDS):
        present = index.tf_matrix(terms, f) > 0        # N x |q| Boolean
        if mode == "fraction":
            S[:, j] = present.mean(axis=1)
        elif mode == "idf":
            idf = np.array([index.idf(t) for t in terms])
            S[:, j] = present @ idf / (idf.sum() or 1.0)
        elif mode == "any":
            S[:, j] = present.any(axis=1)
        elif mode == "all":
            S[:, j] = present.all(axis=1)
        else:
            raise ValueError(mode)
    return S


def weight_grid(step=0.05, n=len(FIELDS)):
    """All weight vectors on the simplex whose entries are multiples of step."""
    m = int(round(1 / step))
    grid = [np.array(c) / m
            for c in itertools.product(range(m + 1), repeat=n) if sum(c) == m]
    return np.array(grid)


def _ap_matrix(scores, rel_mask, n_rel):
    """AP of every column of scores (N x G) given a Boolean relevance mask."""
    # stable sort on -score => ties are broken by document order (doc id)
    order = np.argsort(-scores, axis=0, kind="stable")
    rel_sorted = rel_mask[order]                         # N x G
    hits = np.cumsum(rel_sorted, axis=0)
    ranks = np.arange(1, scores.shape[0] + 1)[:, None]
    return (rel_sorted * hits / ranks).sum(axis=0) / n_rel


class ZoneScorer:
    def __init__(self, index, queries, qrels, mode="fraction"):
        self.index, self.queries, self.qrels, self.mode = index, queries, qrels, mode
        self._cache = {}

    def features(self, q):
        if q not in self._cache:
            self._cache[q] = zone_features(self.index, self.queries[q], self.mode)
        return self._cache[q]

    def _rel_mask(self, q):
        mask = np.zeros(self.index.N, dtype=bool)
        for d in self.qrels.get(q, {}):
            mask[self.index.pos[d]] = True
        return mask

    def grid_search(self, train_q, step=0.05, objective="MAP"):
        """Return (best weights, best objective value, all grid values).

        objective = "MAP": maximise mean average precision on train queries.
        objective = "MSE": minimise the squared error between score and the
            binary relevance label (Manning et al., IIR sec. 6.1.2), with the
            relevant and non-relevant classes weighted equally.
        """
        G = weight_grid(step)
        if objective == "MAP":
            total = np.zeros(len(G))
            for q in train_q:
                mask = self._rel_mask(q)
                if not mask.any():
                    continue
                total += _ap_matrix(np.round(self.features(q) @ G.T, 10), mask, mask.sum())
            values = total / len(train_q)
            best = int(np.argmax(values))
        elif objective == "MSE":
            err_pos, err_neg, n_pos, n_neg = 0, 0, 0, 0
            for q in train_q:
                mask = self._rel_mask(q)
                pred = self.features(q) @ G.T
                err_pos = err_pos + ((1 - pred[mask]) ** 2).sum(axis=0)
                err_neg = err_neg + (pred[~mask] ** 2).sum(axis=0)
                n_pos += mask.sum()
                n_neg += (~mask).sum()
            values = 0.5 * (err_pos / n_pos + err_neg / n_neg)
            best = int(np.argmin(values))
        else:
            raise ValueError(objective)
        return G[best], float(values[best]), (G, values)

    def score(self, q, weights):
        # rounding removes floating-point noise, so that documents with equal
        # Boolean evidence really tie (and are then ordered by doc id)
        return np.round(self.features(q) @ weights, 10)

    def map_of(self, qs, weights):
        aps = []
        for q in qs:
            s = self.score(q, weights)
            order = np.argsort(-s, kind="stable")
            ranked = [self.index.doc_ids[i] for i in order]
            aps.append(average_precision(ranked, self.qrels.get(q, {})))
        return float(np.mean(aps))
