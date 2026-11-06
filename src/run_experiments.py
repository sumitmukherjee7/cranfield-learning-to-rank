"""
Runs every experiment of the assignment and writes the outputs to results/.

    python src/run_experiments.py

Outputs
    results/methodA_test_ranking.txt        Method A+ (final, idf-weighted zone match)
    results/methodB_test_ranking.txt        Method B+ (final, improved linear SVM)
    results/methodA_basic_test_ranking.txt  Method A  (fraction zone match)
    results/methodB_basic_test_ranking.txt  Method B  (the 22 specified features)
    results/baseline_*_test_ranking.txt     BM25 / BM25+PRF / VSM baselines
    results/results.json                    every number used in the report
    results/per_query_test.csv              per-query metrics of the final systems
"""
import itertools
import json
import os
import time

import numpy as np
from scipy import stats

from cranfield import (FIELDS, RESULTS_DIR, CranfieldIndex, evaluate_run,
                       interpolated_pr_curve, kfold_queries, parse_documents,
                       parse_qrels, parse_queries, query_terms, rank,
                       split_queries, write_trec_run)
from features import FEATURE_NAMES, FeatureExtractor, feature_names
from method_a_zone import ZoneScorer, weight_grid
from method_b_svm import SVMRanker

C_GRID = [0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]
STEP = 0.05
NEG = dict(n_top_neg=100, n_rand_neg=200)          # pairwise negative sampling
K1_GRID, B_GRID = [0.6, 0.9, 1.2, 1.5, 2.0, 2.5], [0.3, 0.5, 0.75, 0.9]
PRF_GRID = list(itertools.product([5, 10, 15], [20, 40, 60], [0.2, 0.3, 0.5]))
EXTRAS = ("prf", "title_bm25", "body_bm25")

UNI, VSM = "Boolean zone (uniform)", "VSM cosine"
BM25, BM25T, PRF = "BM25 (k1=1.2, b=0.75)", "BM25 (tuned)", "BM25 + PRF (tuned)"
SA, SAP = "Method A (fraction match)", "Method A+ (idf-weighted match)"
SB, SBP = "Method B (22 features)", "Method B+ (improved)"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def make_run(score_fn, qs, doc_ids):
    run, scores = {}, {}
    for q in qs:
        run[q], scores[q] = rank(score_fn(q), doc_ids)
    return run, scores


def r4(d):
    return {k: round(float(v), 4) for k, v in d.items()}


class Ctx:
    """Shared data so helper functions can be reused inside the CV loop."""

    def __init__(self):
        self.docs, self.Q, self.R = parse_documents(), parse_queries(), parse_qrels()
        self.ix = CranfieldIndex(self.docs)

    def map_of(self, fn, qs):
        run, _ = make_run(fn, qs, self.ix.doc_ids)
        return evaluate_run(run, self.R)[0]["MAP"]

    def test(self, fn, qs, cutoffs=(5, 10, 20)):
        run, sc = make_run(fn, qs, self.ix.doc_ids)
        m, pq = evaluate_run(run, self.R, cutoffs)
        return run, sc, m, pq

    # -- tuning of the BM25 / PRF parameters on training queries --------
    def tune_bm25(self, qs, k1s=K1_GRID, bs=B_GRID):
        res = {}
        for k1 in k1s:
            for b in bs:
                ex = FeatureExtractor(self.ix, k1=k1, b=b)
                res[(k1, b)] = self.map_of(lambda q: ex.bm25(query_terms(self.Q[q])), qs)
        best = max(res, key=res.get)
        return best, res

    def tune_prf(self, qs, k1, b, grid=PRF_GRID):
        res = {}
        for nd, nt, lam in grid:
            ex = FeatureExtractor(self.ix, k1=k1, b=b, prf_docs=nd, prf_terms=nt,
                                  prf_lambda=lam)
            res[(nd, nt, lam)] = self.map_of(lambda q: ex.prf_bm25(query_terms(self.Q[q])), qs)
        best = max(res, key=res.get)
        return best, res

    # -- SVM model selection: formulation x C by 5-fold CV ----------------
    def select_svm(self, ex, qs, modes=("pointwise", "pairwise"), Cs=C_GRID, seed=1):
        folds = kfold_queries(qs, k=5, seed=seed)
        cv = {}
        for mode in modes:
            kw = NEG if mode == "pairwise" else {}
            for C in Cs:
                maps = []
                for val in folds:
                    fit = [q for q in qs if q not in set(val)]
                    s = SVMRanker(ex, self.Q, self.R, ("all",), "query", mode, C, **kw).fit(fit)
                    maps.append(self.map_of(s.score, val))
                cv[(mode, C)] = float(np.mean(maps))
        best = max(cv, key=cv.get)
        return best, cv

    def fit_svm(self, ex, qs, mode, C):
        kw = NEG if mode == "pairwise" else {}
        return SVMRanker(ex, self.Q, self.R, ("all",), "query", mode, C, **kw).fit(qs)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    c = Ctx()
    Q, R, ix = c.Q, c.R, c.ix
    all_q = sorted(Q)
    train_q, test_q = split_queries(all_q, test_fraction=0.3, seed=42)
    log(f"{ix.N} docs, {len(Q)} queries, train={len(train_q)} test={len(test_q)}")

    out = {"data": {
        "n_docs": ix.N, "n_queries": len(Q), "vocab": len(ix.df),
        "n_train": len(train_q), "n_test": len(test_q), "test_queries": test_q,
        "rel_pairs": sum(len(v) for v in R.values()),
        "avg_rel_per_query": float(np.mean([len(R[q]) for q in all_q])),
        "avg_query_terms": float(np.mean([len(query_terms(Q[q])) for q in all_q])),
        "avg_len": {f: float(ix.avg_len[f]) for f in ix.avg_len},
        "grade_counts": {str(g): sum(1 for q in R for d in R[q] if R[q][d] == g)
                         for g in (4, 3, 2, 1)},
    }}

    # ==================================================================
    # Method A  (zone match definitions x objectives)
    # ==================================================================
    A = {"configs": []}
    scorers, best_w, landscape = {}, {}, {}
    for mode in ["fraction", "idf", "any", "all"]:
        zs = ZoneScorer(ix, Q, R, mode)
        scorers[mode] = zs
        for objective in ["MAP", "MSE"]:
            w, val, (G, vals) = zs.grid_search(train_q, STEP, objective)
            _, _, m, _ = c.test(lambda q: zs.score(q, w), test_q)
            A["configs"].append({"mode": mode, "objective": objective,
                                 "weights": dict(zip(FIELDS, map(float, w))),
                                 "train_objective": round(val, 4),
                                 "train_MAP": round(zs.map_of(train_q, w), 4),
                                 "test": r4(m)})
            log(f"A {mode}/{objective}: w={w} test MAP={m['MAP']:.4f}")
            if objective == "MAP":
                best_w[mode], landscape[mode] = w, (G, vals)

    zs, zsi = scorers["fraction"], scorers["idf"]
    wA, wAP = best_w["fraction"], best_w["idf"]
    A["step"] = []
    for mode in ["fraction", "idf"]:
        for step in [0.25, 0.1, 0.05]:
            w, val, _ = scorers[mode].grid_search(train_q, step, "MAP")
            A["step"].append({"mode": mode, "step": step, "n_points": len(weight_grid(step)),
                              "weights": dict(zip(FIELDS, map(float, w))),
                              "train_MAP": round(val, 4),
                              "test_MAP": round(scorers[mode].map_of(test_q, w), 4)})
            log(f"A {mode} step {step}: {w} train {val:.4f}")
    A["fixed"] = []
    for mode in ["fraction", "idf"]:
        for name, w in [("title only", [1, 0, 0, 0]), ("author only", [0, 1, 0, 0]),
                        ("biblio only", [0, 0, 1, 0]), ("body only", [0, 0, 0, 1]),
                        ("uniform", [.25, .25, .25, .25])]:
            w = np.array(w, float)
            _, _, m, _ = c.test(lambda q: scorers[mode].score(q, w), test_q)
            A["fixed"].append({"mode": mode, "name": name, "test": r4(m),
                               "train_MAP": round(scorers[mode].map_of(train_q, w), 4)})
    A["landscape"], A["top_grid_points"] = {}, {}
    for mode in ["fraction", "idf"]:
        G, vals = landscape[mode]
        A["landscape"][mode] = {"grid": G.tolist(), "train_MAP": vals.tolist()}
        top = np.argsort(-vals)[:10]
        A["top_grid_points"][mode] = [{"weights": G[i].tolist(),
                                       "train_MAP": round(float(vals[i]), 4)} for i in top]
    A["best_weights"] = {m: dict(zip(FIELDS, map(float, best_w[m]))) for m in ["fraction", "idf"]}
    A["zone_match_rate"] = {}
    for f_i, f in enumerate(FIELDS):
        rel_rates, non_rates = [], []
        for q in train_q:
            S = zs.features(q)[:, f_i]
            mask = np.zeros(ix.N, bool)
            mask[[ix.pos[d] for d in R[q]]] = True
            rel_rates.append((S[mask] > 0).mean())
            non_rates.append((S[~mask] > 0).mean())
        A["zone_match_rate"][f] = {"relevant": float(np.mean(rel_rates)),
                                   "non_relevant": float(np.mean(non_rates))}
    out["method_A"] = A

    # ==================================================================
    # Method B  (the 22 specified features)
    # ==================================================================
    ex = FeatureExtractor(ix, zs, wA)
    B = {}
    (modeB, CB), cvB = c.select_svm(ex, train_q)
    log(f"B select: {modeB} C={CB} cv={max(cvB.values()):.4f}")
    B["cv"] = [{"mode": m, "C": C_, "cv_MAP": round(v, 4)} for (m, C_), v in cvB.items()]
    B["variants"] = []
    models = {}
    for mode in ["pointwise", "pairwise"]:
        C_ = max((k for k in cvB if k[0] == mode), key=cvB.get)[1]
        s = c.fit_svm(ex, train_q, mode, C_)
        run, sc, m, pq = c.test(s.score, test_q)
        models[mode] = (s, run, sc)
        B["variants"].append({"mode": mode, "C": C_, "cv_MAP": round(cvB[(mode, C_)], 4),
                              "test": r4(m)})
        log(f"B {mode}: C={C_} test MAP={m['MAP']:.4f}")
    sB = models[modeB][0]
    B["main_mode"], B["main_C"] = modeB, CB
    B["weights"] = dict(zip(FEATURE_NAMES, map(float, sB.w)))

    B["ablations"] = []

    def ablate(name, mode=modeB, C=CB, **kw):
        if mode == "pairwise":
            kw = {**NEG, **kw}
        s = SVMRanker(ex, Q, R, kw.pop("streams", ("all",)), kw.pop("norm", "query"),
                      mode, C, **kw).fit(train_q)
        _, _, m, _ = c.test(s.score, test_q)
        B["ablations"].append({"name": name, "n_features": len(s.w), "test": r4(m)})
        log(f"B ablation {name}: MAP={m['MAP']:.4f}")
        return s

    ablate("global z-score normalisation", norm="global")
    ablate("+ title and body streams (64 features)", streams=("all", "title", "body"))
    C_pair = models["pairwise"][0].C
    ablate(f"pairwise (C={C_pair}), 100 hard + 200 random negatives", "pairwise", C_pair)
    ablate(f"pairwise (C={C_pair}), 50 hard + 50 random negatives", "pairwise", C_pair,
           n_top_neg=50, n_rand_neg=50)
    ablate(f"pairwise (C={C_pair}), 200 hard negatives only", "pairwise", C_pair,
           n_top_neg=200, n_rand_neg=0)
    for name, cols in {"without BM25/VSM/Boolean (features 1-19)": list(range(19)),
                       "without TF*IDF stats (15-19)": [i for i in range(22) if not 14 <= i <= 18],
                       "only 20-22 (Boolean, VSM, BM25)": [19, 20, 21]}.items():
        ablate(name, columns=cols)

    # ==================================================================
    # Improvements (all parameters chosen on the training queries only)
    # ==================================================================
    I = {}
    (k1, b), bm_grid = c.tune_bm25(train_q)
    I["bm25_grid"] = [{"k1": k, "b": bb, "train_MAP": round(v, 4)} for (k, bb), v in bm_grid.items()]
    I["bm25_best"] = {"k1": k1, "b": b, "train_MAP": round(bm_grid[(k1, b)], 4),
                      "default_train_MAP": round(bm_grid[(1.2, 0.75)], 4)}
    log(f"tuned BM25 k1={k1} b={b}")
    (nd, nt, lam), prf_grid = c.tune_prf(train_q, k1, b)
    I["prf_grid"] = [{"docs": a, "terms": t, "lambda": l, "train_MAP": round(v, 4)}
                     for (a, t, l), v in prf_grid.items()]
    I["prf_best"] = {"docs": nd, "terms": nt, "lambda": lam,
                     "train_MAP": round(prf_grid[(nd, nt, lam)], 4)}
    log(f"tuned PRF docs={nd} terms={nt} lambda={lam}")
    prf_kw = dict(prf_docs=nd, prf_terms=nt, prf_lambda=lam)

    # improvement ladder for Method B: each step adds one change
    ladder_defs = [
        ("22 features (Method B)", dict(zs=zs, w=wA)),
        ("+ idf-weighted Boolean feature (20)", dict(zs=zsi, w=wAP)),
        ("+ tuned BM25 parameters", dict(zs=zsi, w=wAP, k1=k1, b=b)),
        ("+ PRF BM25 feature (23 features)", dict(zs=zsi, w=wAP, k1=k1, b=b, extras=("prf",), **prf_kw)),
        ("+ title and body BM25 (25 features)", dict(zs=zsi, w=wAP, k1=k1, b=b, extras=EXTRAS, **prf_kw)),
    ]
    I["ladder"] = []
    best_step, best_cv = None, -1
    for name, d in ladder_defs:
        d = dict(d)
        exs = FeatureExtractor(ix, d.pop("zs"), d.pop("w"), **d)
        (mode, C_), cv = c.select_svm(exs, train_q)
        s = c.fit_svm(exs, train_q, mode, C_)
        run, sc, m, pq = c.test(s.score, test_q)
        I["ladder"].append({"name": name, "mode": mode, "C": C_, "n_features": len(s.w),
                            "cv_MAP": round(cv[(mode, C_)], 4), "test": r4(m)})
        log(f"ladder {name}: {mode} C={C_} cv={cv[(mode, C_)]:.4f} test MAP={m['MAP']:.4f}")
        if cv[(mode, C_)] > best_cv:
            best_cv, best_step = cv[(mode, C_)], (name, exs, s, run, sc, d)
    nameBP, exP, sBP, runBP, scBP, dBP = best_step
    I["final_B"] = {"step": nameBP, "mode": sBP.mode, "C": sBP.C,
                    "extras": list(exP.extras)}
    I["weights_B_plus"] = dict(zip(feature_names(extras=exP.extras), map(float, sBP.w)))
    I["A_plus_weights"] = A["best_weights"]["idf"]
    out["improved"] = I
    out["method_B"] = B
    log(f"final Method B+ = {nameBP}")

    # ==================================================================
    # Final comparison on the test queries
    # ==================================================================
    exT = FeatureExtractor(ix, k1=k1, b=b, **prf_kw)
    raw = lambda q: ex.features(q, Q[q])
    sysfn = {
        UNI: lambda q: zs.score(q, np.full(4, .25)),
        VSM: lambda q: raw(q)[:, 20],
        BM25: lambda q: raw(q)[:, 21],
        BM25T: lambda q: exT.bm25(query_terms(Q[q])),
        PRF: lambda q: exT.prf_bm25(query_terms(Q[q])),
        SA: lambda q: zs.score(q, wA),
        SAP: lambda q: zsi.score(q, wAP),
        SB: sB.score,
        SBP: sBP.score,
    }
    runs = {n: make_run(f, test_q, ix.doc_ids) for n, f in sysfn.items()}
    for fname, n, tag in [("methodA_test_ranking.txt", SAP, "MethodA_plus_ZoneBoolean_idf"),
                          ("methodB_test_ranking.txt", SBP, f"MethodB_plus_LinearSVM_{sBP.mode}"),
                          ("methodA_basic_test_ranking.txt", SA, "MethodA_ZoneBoolean"),
                          ("methodB_basic_test_ranking.txt", SB, f"MethodB_LinearSVM_{modeB}"),
                          ("baseline_bm25_test_ranking.txt", BM25, "BM25"),
                          ("baseline_bm25_prf_test_ranking.txt", PRF, "BM25_PRF"),
                          ("baseline_vsm_test_ranking.txt", VSM, "VSM")]:
        write_trec_run(os.path.join(RESULTS_DIR, fname), *runs[n], tag)

    comp, per_query, curves, by_k = {}, {}, {}, {}
    ks = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]
    for n, (run, _) in runs.items():
        m, pq = evaluate_run(run, R, cutoffs=(1, 5, 10, 20, 50))
        comp[n], per_query[n] = r4(m), pq
        lv, pr = interpolated_pr_curve(run, R)
        curves[n] = pr.tolist()
        mk, _ = evaluate_run(run, R, cutoffs=ks)
        by_k[n] = {"k": ks, "P": [mk[f"P@{k}"] for k in ks],
                   "R": [mk[f"R@{k}"] for k in ks], "F1": [mk[f"F1@{k}"] for k in ks]}
    out["comparison"] = comp
    out["pr_curves"] = {"recall": lv.tolist(), "precision": curves}
    out["prf_by_k"] = by_k
    out["names"] = {"UNI": UNI, "VSM": VSM, "BM25": BM25, "BM25T": BM25T, "PRF": PRF,
                    "A": SA, "AP": SAP, "B": SB, "BP": SBP}

    sig = []
    for a, b_ in [(SB, SA), (SAP, SA), (SBP, SB), (SBP, SAP), (SB, BM25), (SBP, BM25),
                  (SBP, PRF), (SA, UNI)]:
        for metric in ["MAP", "F1@10"]:
            x = np.array([per_query[a][q][metric] for q in test_q])
            y = np.array([per_query[b_][q][metric] for q in test_q])
            t = stats.ttest_rel(x, y)
            try:
                wpv = stats.wilcoxon(x, y).pvalue
            except ValueError:
                wpv = 1.0
            sig.append({"a": a, "b": b_, "metric": metric,
                        "mean_a": round(float(x.mean()), 4), "mean_b": round(float(y.mean()), 4),
                        "t_p": round(float(t.pvalue), 4), "wilcoxon_p": round(float(wpv), 4),
                        "wins": int((x > y).sum()), "losses": int((x < y).sum())})
    out["significance"] = sig

    with open(os.path.join(RESULTS_DIR, "per_query_test.csv"), "w") as fh:
        keys = ["P@10", "R@10", "F1@10", "MAP"]
        cols = [("A", SA), ("Aplus", SAP), ("B", SB), ("Bplus", SBP)]
        fh.write("query," + ",".join(f"{p}_{k}" for p, _ in cols for k in keys) + ",n_relevant\n")
        for q in test_q:
            fh.write(f"{q}," + ",".join(f"{per_query[n][q][k]:.4f}" for _, n in cols for k in keys)
                     + f",{len(R[q])}\n")
    log("comparison done")

    # ==================================================================
    # 5-fold cross-validation over all 225 queries.  Every learned
    # parameter (zone weights, BM25 k1/b, PRF settings, SVM weights) is
    # re-learned on the training part of each fold; the SVM formulation
    # and C are the ones selected on the main training split.
    # ==================================================================
    folds = kfold_queries(all_q, k=5, seed=7)
    cvres = {n: [] for n in ["Method A", "Method A+", "Method B", "Method B+",
                             "BM25", "BM25 + PRF (tuned)"]}
    for i, te in enumerate(folds):
        tr = [q for q in all_q if q not in set(te)]
        za, zi = ZoneScorer(ix, Q, R, "fraction"), ZoneScorer(ix, Q, R, "idf")
        za._cache, zi._cache = zs._cache, zsi._cache      # zone features are query-only
        wa, _, _ = za.grid_search(tr, STEP, "MAP")
        wi, _, _ = zi.grid_search(tr, STEP, "MAP")
        (fk1, fb), _ = c.tune_bm25(tr, [0.9, 1.2, 1.5, 2.0, 2.5], [0.5, 0.75, 0.9])
        (fnd, fnt, flam), _ = c.tune_prf(tr, fk1, fb,
                                         list(itertools.product([5, 10], [20, 40], [0.3, 0.5])))
        fprf = dict(prf_docs=fnd, prf_terms=fnt, prf_lambda=flam)
        exa = FeatureExtractor(ix, za, wa)
        exp_ = FeatureExtractor(ix, zi, wi, k1=fk1, b=fb, extras=exP.extras, **fprf)
        sb = c.fit_svm(exa, tr, modeB, CB)
        sbp = c.fit_svm(exp_, tr, sBP.mode, sBP.C)
        for name, fn in [("Method A", lambda q: za.score(q, wa)),
                         ("Method A+", lambda q: zi.score(q, wi)),
                         ("Method B", sb.score), ("Method B+", sbp.score),
                         ("BM25", lambda q: exa.features(q, Q[q])[:, 21]),
                         ("BM25 + PRF (tuned)", lambda q: exp_.prf_bm25(query_terms(Q[q])))]:
            _, _, m, _ = c.test(fn, te)
            cvres[name].append(r4(m))
        log("CV fold %d: " % (i + 1) + " ".join(f"{n}={v[-1]['MAP']}" for n, v in cvres.items()))
    out["cv5"] = {name: {"folds": v,
                         "mean": r4({k: np.mean([f[k] for f in v]) for k in v[0]}),
                         "std": r4({k: np.std([f[k] for f in v]) for k in v[0]})}
                  for name, v in cvres.items()}

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    log("wrote results/results.json")


if __name__ == "__main__":
    main()
