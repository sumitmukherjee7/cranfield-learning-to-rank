"""
Shared utilities for the Cranfield learning-to-rank assignment.

  * parsing of cran.all.1400 / cran.qry / cranqrel
  * text preprocessing (tokenise, stop-word removal, Porter stemming)
  * an in-memory inverted index with per-field (zone) statistics
  * evaluation measures (P@k, R@k, F1@k, AP, nDCG@k)
  * train / test split of the 225 queries
"""
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
from nltk.stem import PorterStemmer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "cran")
RESULTS_DIR = os.path.join(ROOT, "results")

FIELDS = ["title", "author", "biblio", "body"]          # .T .A .B .W
_MARKERS = {".T": "title", ".A": "author", ".B": "biblio", ".W": "body"}

# A compact English stop-word list (SMART-style subset).
STOPWORDS = set("""
a about above after again against all also am an and any are as at be because
been before being below between both but by can could did do does doing down
during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself
no nor not now of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours
yourself yourselves may might must shall upon within without whether etc per
one two three made make given use used using obtained found known show shown
""".split())

_stemmer = PorterStemmer()
_stem_cache = {}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lower-case, split on non-alphanumerics, drop stop words, Porter-stem."""
    out = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS or len(tok) < 2 and not tok.isdigit():
            continue
        s = _stem_cache.get(tok)
        if s is None:
            s = _stemmer.stem(tok)
            _stem_cache[tok] = s
        out.append(s)
    return out


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def parse_documents(path=None):
    """Return {doc_id: {field: raw_text}} for the 1400 Cranfield documents."""
    path = path or os.path.join(DATA_DIR, "cran.all.1400")
    docs, cur, field = {}, None, None
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(".I "):
                cur = int(line.split()[1])
                docs[cur] = {f: "" for f in FIELDS}
                field = None
            elif line.strip() in _MARKERS:
                field = _MARKERS[line.strip()]
            elif cur is not None and field is not None:
                docs[cur][field] += line + " "
    return docs


def parse_queries(path=None):
    """Return {query_no: text}.

    The .I numbers in cran.qry are not contiguous (001, 002, 004, 008, ...)
    whereas cranqrel numbers queries 1..225 sequentially, so queries are
    re-numbered by their position in the file.
    """
    path = path or os.path.join(DATA_DIR, "cran.qry")
    queries, texts = {}, []
    with open(path, encoding="latin-1") as fh:
        cur = None
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(".I "):
                if cur is not None:
                    texts.append(cur)
                cur = ""
            elif line.strip() == ".W":
                continue
            else:
                cur += line + " "
        if cur is not None:
            texts.append(cur)
    for i, t in enumerate(texts, start=1):
        queries[i] = t.strip()
    return queries


def parse_qrels(path=None):
    """Return {query_no: {doc_id: grade}} with grade = 5 - code (4 = best).

    Cleverdon codes 1..4 are relevant (1 = complete answer, 4 = minimum
    interest).  Each query also carries one pair with code -1; these are not
    relevance judgements and are discarded.
    """
    path = path or os.path.join(DATA_DIR, "cranqrel")
    qrels = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            q, d, code = int(parts[0]), int(parts[1]), int(parts[2])
            if 1 <= code <= 4:
                qrels[q][d] = 5 - code
    return dict(qrels)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------
class CranfieldIndex:
    """Term statistics for the whole document and for every zone."""

    def __init__(self, docs):
        self.doc_ids = sorted(docs)
        self.N = len(self.doc_ids)
        self.pos = {d: i for i, d in enumerate(self.doc_ids)}
        # tokens per zone; the "all" stream is the concatenation of all zones
        self.zone_tf = {f: [] for f in FIELDS + ["all"]}
        self.zone_len = {f: np.zeros(self.N) for f in FIELDS + ["all"]}
        for i, d in enumerate(self.doc_ids):
            all_toks = []
            for f in FIELDS:
                toks = tokenize(docs[d][f])
                all_toks += toks
                self.zone_tf[f].append(Counter(toks))
                self.zone_len[f][i] = len(toks)
            self.zone_tf["all"].append(Counter(all_toks))
            self.zone_len["all"][i] = len(all_toks)

        self.df = Counter()
        for c in self.zone_tf["all"]:
            self.df.update(c.keys())
        self.avg_len = {f: self.zone_len[f].mean() for f in self.zone_len}

        # postings: term -> (array of doc positions, array of tf) per stream
        self.postings = {}
        for f in self.zone_tf:
            p = defaultdict(lambda: ([], []))
            for i, c in enumerate(self.zone_tf[f]):
                for t, n in c.items():
                    p[t][0].append(i)
                    p[t][1].append(n)
            self.postings[f] = {t: (np.array(a), np.array(b, dtype=float))
                                for t, (a, b) in p.items()}

    def idf(self, term):
        df = self.df.get(term, 0)
        return math.log(self.N / df) if df else 0.0

    def tf_matrix(self, terms, stream="all"):
        """Dense (N x |terms|) matrix of raw term frequencies in a stream."""
        m = np.zeros((self.N, len(terms)))
        post = self.postings[stream]
        for j, t in enumerate(terms):
            if t in post:
                idx, tf = post[t]
                m[idx, j] = tf
        return m


def query_terms(text):
    """Unique query terms, in first-occurrence order."""
    seen, out = set(), []
    for t in tokenize(text):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --------------------------------------------------------------------------
# Train / test split
# --------------------------------------------------------------------------
def split_queries(qids, test_fraction=0.3, seed=42):
    qids = sorted(qids)
    rng = random.Random(seed)
    shuffled = qids[:]
    rng.shuffle(shuffled)
    n_test = int(round(len(qids) * test_fraction))
    test = sorted(shuffled[:n_test])
    train = sorted(shuffled[n_test:])
    return train, test


def kfold_queries(qids, k=5, seed=7):
    qids = sorted(qids)
    rng = random.Random(seed)
    shuffled = qids[:]
    rng.shuffle(shuffled)
    return [sorted(shuffled[i::k]) for i in range(k)]


# --------------------------------------------------------------------------
# Ranking / evaluation
# --------------------------------------------------------------------------
def rank(scores, doc_ids):
    """Sort by descending score; ties are broken by ascending doc id."""
    order = np.argsort(-scores, kind="stable")
    return [doc_ids[i] for i in order], scores[order]


def precision_recall_f1_at_k(ranked, rel, k):
    top = ranked[:k]
    hits = sum(1 for d in top if d in rel)
    p = hits / k
    r = hits / len(rel) if rel else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def average_precision(ranked, rel):
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            hits += 1
            s += hits / i
    return s / len(rel) if rel else 0.0


def ndcg_at_k(ranked, rel, k):
    gains = [(2 ** rel.get(d, 0) - 1) / math.log2(i + 2)
             for i, d in enumerate(ranked[:k])]
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return sum(gains) / idcg if idcg else 0.0


CUTOFFS = (5, 10, 20)


def evaluate_run(run, qrels, cutoffs=CUTOFFS):
    """run: {qid: ranked doc list}.  Returns (mean metrics, per-query metrics)."""
    per_q = {}
    for q, ranked in run.items():
        rel = qrels.get(q, {})
        m = {}
        for k in cutoffs:
            p, r, f = precision_recall_f1_at_k(ranked, rel, k)
            m[f"P@{k}"], m[f"R@{k}"], m[f"F1@{k}"] = p, r, f
        # R-precision: cutoff = number of relevant documents (P = R = F1)
        m["R-Prec"] = precision_recall_f1_at_k(ranked, rel, max(len(rel), 1))[0]
        m["MAP"] = average_precision(ranked, rel)
        m["nDCG@10"] = ndcg_at_k(ranked, rel, 10)
        per_q[q] = m
    keys = next(iter(per_q.values())).keys()
    mean = {k: float(np.mean([per_q[q][k] for q in per_q])) for k in keys}
    return mean, per_q


def interpolated_pr_curve(run, qrels, levels=np.linspace(0, 1, 11)):
    """11-point interpolated precision-recall curve averaged over queries."""
    curves = []
    for q, ranked in run.items():
        rel = qrels.get(q, {})
        if not rel:
            continue
        hits, prec, rec = 0, [], []
        for i, d in enumerate(ranked, start=1):
            if d in rel:
                hits += 1
                prec.append(hits / i)
                rec.append(hits / len(rel))
        prec, rec = np.array(prec), np.array(rec)
        curves.append([prec[rec >= l].max() if (rec >= l).any() else 0.0
                       for l in levels])
    return levels, np.mean(curves, axis=0)


def write_trec_run(path, run, scores, tag, depth=100):
    """Write rankings as  qid Q0 docid rank score tag  (TREC format)."""
    with open(path, "w") as fh:
        fh.write(f"# query_id Q0 doc_id rank score run_tag   ({tag}, top {depth})\n")
        for q in sorted(run):
            for r, (d, s) in enumerate(zip(run[q][:depth], scores[q][:depth]), 1):
                fh.write(f"{q} Q0 {d} {r} {s:.6f} {tag}\n")
