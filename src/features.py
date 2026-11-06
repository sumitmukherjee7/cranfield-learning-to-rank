"""
Query-document features for Method B (LETOR-style feature set).

For a query q with unique terms t_1..t_m and a document stream d:

  1  query term number        number of query terms that occur in d
  2  query term ratio         (1) / m
  3  stream length            number of tokens in d
  4  IDF                      sum_i idf(t_i),  idf = log(N / df)
  5-9   sum/min/max/mean/var of tf(t_i, d)
  10-14 sum/min/max/mean/var of tf(t_i, d) / |d|
  15-19 sum/min/max/mean/var of tf(t_i, d) * idf(t_i)
  20 Boolean model score      Boolean weighted zone score (Method A)
  21 vector space model score cosine( ltc query , ltc document )
  22 BM25 score               Okapi BM25, k1 = 1.2, b = 0.75
"""
import math

import numpy as np

from cranfield import query_terms

FEATURE_NAMES = [
    "query term number", "query term ratio", "stream length", "IDF",
    "sum TF", "min TF", "max TF", "mean TF", "var TF",
    "sum TF/len", "min TF/len", "max TF/len", "mean TF/len", "var TF/len",
    "sum TF*IDF", "min TF*IDF", "max TF*IDF", "mean TF*IDF", "var TF*IDF",
    "Boolean model", "VSM cosine", "BM25",
]

K1, B = 1.2, 0.75


EXTRA_NAMES = {
    "prf": "BM25 + pseudo-relevance feedback",
    "title_bm25": "BM25 on title",
    "body_bm25": "BM25 on body",
}


class FeatureExtractor:
    def __init__(self, index, zone_scorer=None, zone_weights=None, k1=K1, b=B,
                 extras=(), prf_docs=10, prf_terms=20, prf_lambda=0.5):
        self.index = index
        self.zone_scorer = zone_scorer
        self.zone_weights = zone_weights
        self.k1, self.b = k1, b
        self.extras = tuple(extras)
        self.prf_docs, self.prf_terms, self.prf_lambda = prf_docs, prf_terms, prf_lambda
        self._fcache = {}
        # Euclidean norms of ltc document vectors, one array per stream
        self.doc_norm = {}
        for stream, tfs in index.zone_tf.items():
            norms = np.zeros(index.N)
            for i, c in enumerate(tfs):
                norms[i] = math.sqrt(sum(((1 + math.log(n)) * index.idf(t)) ** 2
                                         for t, n in c.items()))
            norms[norms == 0] = 1.0
            self.doc_norm[stream] = norms

    def stream_block(self, terms, stream):
        """The 21 stream features (all except the Boolean model score)."""
        ix = self.index
        N = ix.N
        m = len(terms)
        if m == 0:
            return np.zeros((N, 21))
        tf = ix.tf_matrix(terms, stream)                       # N x m
        length = ix.zone_len[stream]
        safe_len = np.where(length > 0, length, 1.0)
        idf = np.array([ix.idf(t) for t in terms])
        ntf = tf / safe_len[:, None]
        tfidf = tf * idf

        def stats(x):
            return [x.sum(1), x.min(1), x.max(1), x.mean(1), x.var(1)]

        covered = (tf > 0).sum(1).astype(float)
        cols = [covered, covered / m, length, np.full(N, idf.sum())]
        cols += stats(tf) + stats(ntf) + stats(tfidf)

        # vector space model: ltc . ltc cosine
        wd = np.where(tf > 0, 1 + np.log(np.maximum(tf, 1)), 0) * idf
        wq = idf.copy()                 # query terms are unique -> log tf = 0
        qnorm = np.linalg.norm(wq) or 1.0
        cols.append(wd @ wq / (self.doc_norm[stream] * qnorm))

        # Okapi BM25
        cols.append(self.bm25(terms, stream=stream, tf=tf))
        return np.column_stack(cols)

    def bm25(self, terms, weights=None, stream="all", tf=None):
        """Okapi BM25 of every document; optional per-term query weights."""
        ix = self.index
        if tf is None:
            tf = ix.tf_matrix(terms, stream)
        df = np.array([ix.df.get(t, 0) for t in terms], dtype=float)
        idf_bm = np.log((ix.N - df + 0.5) / (df + 0.5) + 1)
        if weights is not None:
            idf_bm = idf_bm * np.asarray(weights)
        avg = ix.avg_len[stream] or 1.0
        denom = tf + self.k1 * (1 - self.b + self.b * ix.zone_len[stream][:, None] / avg)
        return (idf_bm * tf * (self.k1 + 1) / denom).sum(1)

    def prf_bm25(self, terms):
        """BM25 with an RM3-style expanded query (pseudo-relevance feedback).

        The top prf_docs documents of the initial BM25 ranking are assumed
        relevant; candidate terms are weighted by sum_d P(t|d) * P(d|q) and
        the prf_terms best are interpolated with the original query:
            w(t) = lambda * [t in q] / |q| + (1 - lambda) * P_fb(t)
        """
        ix = self.index
        s0 = self.bm25(terms)
        top = np.argsort(-s0, kind="stable")[: self.prf_docs]
        p_d = np.exp(s0[top] - s0[top].max())
        p_d /= p_d.sum()
        fb = {}
        for pd, i in zip(p_d, top):
            c = ix.zone_tf["all"][i]
            n = ix.zone_len["all"][i] or 1.0
            for t, f in c.items():
                fb[t] = fb.get(t, 0.0) + pd * f / n
        best = sorted(fb, key=lambda t: (-fb[t], t))[: self.prf_terms]
        z = sum(fb[t] for t in best) or 1.0
        w = {t: self.prf_lambda / len(terms) for t in terms}
        for t in best:
            w[t] = w.get(t, 0.0) + (1 - self.prf_lambda) * fb[t] / z
        exp_terms = list(w)
        return self.bm25(exp_terms, [w[t] for t in exp_terms])

    def boolean_score(self, qid):
        return self.zone_scorer.score(qid, self.zone_weights)

    def features(self, qid, qtext, streams=("all",)):
        """N x F feature matrix.  The first 22 columns follow FEATURE_NAMES
        for the whole-document stream; additional streams (e.g. title, body)
        append 21 further columns each."""
        key = (qid, tuple(streams))
        if key in self._fcache:
            return self._fcache[key]
        terms = query_terms(qtext)
        blocks = [self.stream_block(terms, s) for s in streams]
        main = blocks[0]
        X = np.column_stack([main[:, :19], self.boolean_score(qid), main[:, 19:]])
        extra = []
        for e in self.extras:
            if e == "prf":
                extra.append(self.prf_bm25(terms) if terms else np.zeros(self.index.N))
            elif e == "title_bm25":
                extra.append(self.bm25(terms, stream="title"))
            elif e == "body_bm25":
                extra.append(self.bm25(terms, stream="body"))
            else:
                raise ValueError(e)
        if len(blocks) > 1 or extra:
            X = np.column_stack([X] + blocks[1:] + extra)
        self._fcache[key] = X
        return X


def feature_names(streams=("all",), extras=()):
    names = list(FEATURE_NAMES)
    block = [n for n in FEATURE_NAMES if n != "Boolean model"]
    for s in streams[1:]:
        names += [f"{s}: {n}" for n in block]
    return names + [EXTRA_NAMES[e] for e in extras]


def query_level_minmax(X):
    """LETOR QueryLevelNorm: scale every feature to [0, 1] within a query."""
    lo, hi = X.min(0), X.max(0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / rng
