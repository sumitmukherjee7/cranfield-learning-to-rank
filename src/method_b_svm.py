"""
Method B - real-valued feature weights learned with a linear SVM.

Two ways of turning relevance judgements into an SVM problem are provided:

  pointwise  Every (query, document) pair is one example labelled
             relevant (+1) / non-relevant (-1); a linear SVM with balanced
             class weights is trained and documents are ranked by w . x.

  pairwise   RankSVM (Joachims, 2002).  For two documents d_i, d_j of the
             same query with grade(d_i) > grade(d_j) the difference vector
             x_i - x_j is a positive example (and x_j - x_i a negative one).
             The learned w is again used to rank by w . x.

Graded relevance: grade = 5 - Cleverdon code (4 = complete answer ..
1 = minimum interest), 0 for unjudged documents.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from features import query_level_minmax


class SVMRanker:
    def __init__(self, extractor, queries, qrels, streams=("all",),
                 norm="query", mode="pairwise", C=1.0, n_top_neg=50,
                 n_rand_neg=50, seed=0, columns=None, log=False):
        self.ex, self.queries, self.qrels = extractor, queries, qrels
        self.streams, self.norm, self.mode, self.C = streams, norm, mode, C
        self.n_top_neg, self.n_rand_neg, self.seed = n_top_neg, n_rand_neg, seed
        self.columns = columns          # optional feature subset (ablations)
        self.log = log                  # log(1 + x) before normalisation
        self._cache = {}
        self.scaler = None
        self.w = None

    # -- features ---------------------------------------------------------
    def raw(self, q):
        key = (q, self.streams)
        if key not in self._cache:
            self._cache[key] = self.ex.features(q, self.queries[q], self.streams)
        return self._cache[key]

    def X(self, q):
        X = self.raw(q)
        if self.log:
            X = np.log1p(X)
        if self.norm == "query":
            X = query_level_minmax(X)
        elif self.norm == "global":
            X = self.scaler.transform(X)
        return X if self.columns is None else X[:, self.columns]

    def grades(self, q):
        g = np.zeros(self.ex.index.N)
        for d, r in self.qrels.get(q, {}).items():
            g[self.ex.index.pos[d]] = r
        return g

    # -- training ---------------------------------------------------------
    def _candidates(self, q, X, g, rng):
        """Relevant docs + hardest (highest BM25) and random non-relevant docs."""
        rel = np.flatnonzero(g > 0)
        non = np.flatnonzero(g == 0)
        bm25 = self.raw(q)[:, 21]
        hard = non[np.argsort(-bm25[non], kind="stable")[: self.n_top_neg]]
        rest = np.setdiff1d(non, hard)
        rand = rng.choice(rest, size=min(self.n_rand_neg, len(rest)), replace=False)
        return np.concatenate([rel, hard, rand])

    def fit(self, train_q):
        rng = np.random.default_rng(self.seed)
        if self.norm == "global":
            raw = np.vstack([self.raw(q) for q in train_q])
            self.scaler = StandardScaler().fit(np.log1p(raw) if self.log else raw)

        if self.mode == "pointwise":
            Xs, ys = [], []
            for q in train_q:
                X, g = self.X(q), self.grades(q)
                Xs.append(X)
                ys.append(np.where(g > 0, 1, -1))
            svm = LinearSVC(C=self.C, class_weight="balanced", dual=False,
                            max_iter=20000)
            svm.fit(np.vstack(Xs), np.concatenate(ys))
        elif self.mode == "pairwise":
            diffs = []
            for q in train_q:
                X, g = self.X(q), self.grades(q)
                if not (g > 0).any():
                    continue
                cand = self._candidates(q, X, g, rng)
                Xc, gc = X[cand], g[cand]
                i, j = np.nonzero(gc[:, None] > gc[None, :])
                diffs.append(Xc[i] - Xc[j])
            D = np.vstack(diffs)
            flip = rng.random(len(D)) < 0.5            # balance the two classes
            D[flip] *= -1
            y = np.where(flip, -1, 1)
            svm = LinearSVC(C=self.C, fit_intercept=False, max_iter=20000)
            svm.fit(D, y)
        else:
            raise ValueError(self.mode)
        self.w = svm.coef_.ravel()
        return self

    # -- ranking ----------------------------------------------------------
    def score(self, q):
        return self.X(q) @ self.w
