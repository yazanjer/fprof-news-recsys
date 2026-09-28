"""F-PROF re-ranking and the Chapter 3 evaluation metrics.

Re-ranking objective (per list L with candidate set C, |L| = K):
    max_x  sum_j (r_j - lambda * w_j) x_j   s.t.  sum_j x_j = K,  x_j in {0,1}
with w_j = Y_j (binary popularity indicator) or w_j = P_j (popularity percentile).
Because the objective is linear and the only constraint is the cardinality
constraint, the optimum is the K items with the largest adjusted score
r_j - lambda * w_j; the selected items are presented in descending order of
that adjusted score. CPLEX is used only as an independent cross-check.
"""
import numpy as np
from scipy.stats import rankdata, wilcoxon

K_DEFAULT = 5


# ------------------------------------------------------------------ grouping
def make_groups(test, scores, mode, K):
    """Return dict of row arrays sorted by group, restricted to eligible groups
    (>= K candidates and >= 1 click). mode = 'impression' | 'user'."""
    if mode == "impression":
        g, i, y, r = test["imp"], test["i"], test["y"].astype(np.int8), scores
    else:  # pool all test impressions of a user; label = clicked in any; score = mean
        key = test["user_key"].astype(np.int64) * (1 << 22) + test["i"]
        uk, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv)
        r = np.bincount(inv, weights=scores) / cnt
        y = np.zeros(len(uk), np.int8)
        np.maximum.at(y, inv, test["y"].astype(np.int8))
        g = (uk >> 22).astype(np.int64)
        i = (uk & ((1 << 22) - 1)).astype(np.int64)
    o = np.argsort(g, kind="stable")
    g, i, y, r = g[o], i[o], y[o], r[o]
    size = np.bincount(g)
    pos = np.bincount(g, weights=y)
    ok = (size >= K) & (pos >= 1)
    keep = ok[g]
    g, i, y, r = g[keep], i[keep], y[keep], r[keep]
    _, g = np.unique(g, return_inverse=True)
    return {"g": g, "i": i, "y": y, "r": r, "G": int(g.max() + 1)}


def select_topk(G, adj, K):
    """Indices of the K rows with largest adj in each group, ordered by adj desc,
    together with their 1-based position."""
    g = G["g"]
    o = np.lexsort((-adj, g))
    gs = g[o]
    start = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
    first = np.repeat(start, np.diff(np.r_[start, len(gs)]))
    rank = np.arange(len(gs)) - first
    m = rank < K
    return o[m], rank[m] + 1


# ------------------------------------------------------------------ metrics
def metrics(G, sel, posn, clicks, P, Y, K, alpha=0.5):
    g, it, y = G["g"][sel], G["i"][sel], G["y"][sel].astype(float)
    nG = G["G"]
    v = 1.0 / np.log2(posn + 1)
    hits = np.bincount(g, weights=y, minlength=nG)
    nrel = np.bincount(G["g"], weights=G["y"].astype(float), minlength=nG)
    dcg = np.bincount(g, weights=y * v, minlength=nG)
    ideal = np.array([np.sum(1.0 / np.log2(np.arange(2, min(n, K) + 2))) for n in range(K + 1)])
    idcg = ideal[np.minimum(nrel, K).astype(int)]
    ndcg = dcg / idcg
    first_hit = np.full(nG, np.inf)
    hp = posn[y > 0]
    np.minimum.at(first_hit, g[y > 0], hp)
    rr = np.where(np.isfinite(first_hit), 1.0 / first_hit, 0.0)

    cat = np.unique(G["i"])                      # test catalogue I (candidate items)
    tail_mask_cat = Y[cat] == 0
    rec_items = np.unique(it)
    exp_cnt = np.bincount(it, minlength=len(Y))[cat].astype(float)
    ee = np.bincount(it, weights=v, minlength=len(Y))[cat] / nG
    is_tail = (Y[it] == 0).astype(float)
    srt = np.sort(exp_cnt)
    n = len(srt)
    gini = (2 * np.sum(np.arange(1, n + 1) * srt) / (n * srt.sum()) - (n + 1) / n) if srt.sum() > 0 else 0.0
    n_pop = np.sum(1 - is_tail)
    out = {
        "P@K": hits.mean() / K, "R@K": np.mean(hits / nrel), "NDCG@K": ndcg.mean(),
        "HR@K": np.mean(hits > 0), "MRR@K": rr.mean(),
        "AD": float(len(rec_items)), "CC": len(rec_items) / len(cat),
        "LTC": np.isin(rec_items, cat[tail_mask_cat]).sum() / max(tail_mask_cat.sum(), 1),
        "ARP_clicks": np.mean(np.bincount(g, weights=clicks[it], minlength=nG) / K),
        "ARP_pct": np.mean(np.bincount(g, weights=P[it], minlength=nG) / K),
        "APLT": np.mean(np.bincount(g, weights=is_tail, minlength=nG) / K),
        "PR": n_pop / max(is_tail.sum(), 1),
        "Gini": float(gini),
        "ED": float(ee[~tail_mask_cat].mean() - ee[tail_mask_cat].mean()) if (~tail_mask_cat).any() and tail_mask_cat.any() else 0.0,
        "NDCE": float(np.sum(is_tail * v) / np.sum(v)),
        "n_lists": nG, "catalogue": int(len(cat)), "tail_catalogue": int(tail_mask_cat.sum()),
    }
    out["CTS"] = alpha * out["NDCG@K"] + (1 - alpha) * (out["APLT"] + out["CC"] + (1 - out["Gini"])) / 3
    return {k: float(val) for k, val in out.items()}, ndcg


def impression_auc(G):
    """Mean per-list AUC over lists that contain both classes (MIND convention)."""
    g, y, r = G["g"], G["y"], G["r"]
    out = []
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1, len(g)]
    for a, b in zip(starts[:-1], starts[1:]):
        yy = y[a:b]
        npos = yy.sum()
        if 0 < npos < len(yy):
            rk = rankdata(r[a:b])
            out.append((rk[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(yy) - npos)))
    return float(np.mean(out))


# ------------------------------------------------------------------ grid
def rerank_grid(test, scores, clicks, P, Y, lambdas, K=K_DEFAULT, keep_ndcg=False):
    rows, ndcgs = [], {}
    for mode in ("impression", "user"):
        G = make_groups(test, scores, mode, K)
        base = None
        for wname, w in (("binary", Y), ("continuous", P)):
            for lam in lambdas:
                if lam == 0 and wname == "continuous":
                    continue
                adj = G["r"] - lam * w[G["i"]]
                sel, posn = select_topk(G, adj, K)
                m, nd = metrics(G, sel, posn, clicks, P, Y, K)
                rows.append({"mode": mode, "w": "none" if lam == 0 else wname, "lambda": lam, **m})
                if keep_ndcg:
                    ndcgs[(mode, "none" if lam == 0 else wname, lam)] = nd
        # original (sign-reversed) objective of the submitted thesis, Eq. (3.38):
        # max sum r x - lam * sum (Y - xY)^2  ==  sum (r + lam*Y^2) x - const
        for wname, w in (("binary", Y), ("continuous", P)):
            adj = G["r"] + 0.5 * w[G["i"]] ** 2
            sel, posn = select_topk(G, adj, K)
            m, _ = metrics(G, sel, posn, clicks, P, Y, K)
            rows.append({"mode": mode, "w": "legacy_" + wname, "lambda": 0.5, **m})
        if mode == "impression":
            auc = impression_auc(G)
            for r_ in rows:
                if r_["mode"] == "impression":
                    r_["imp_AUC"] = auc
    sig = []
    if keep_ndcg:
        for (mode, wname, lam), nd in ndcgs.items():
            if lam == 0:
                continue
            b = ndcgs[(mode, "none", 0.0)]
            d = nd - b
            p = float(wilcoxon(nd, b, zero_method="wilcox").pvalue) if np.any(d != 0) else 1.0
            sig.append({"mode": mode, "w": wname, "lambda": lam, "mean_diff": float(d.mean()),
                        "share_lists_changed": float(np.mean(d != 0)), "wilcoxon_p": p})
    return rows, sig


def cplex_check(test, scores, P, Y, lam=0.5, K=K_DEFAULT, n_sample=2000, seed=0):
    """Solve a random sample of impression lists with DOcplex and compare with the
    sort-based optimum. Every solve status is recorded; there is no fallback."""
    from docplex.mp.model import Model
    G = make_groups(test, scores, "impression", K)
    rng = np.random.default_rng(seed)
    gids = rng.choice(G["G"], size=min(n_sample, G["G"]), replace=False)
    starts = np.r_[0, np.flatnonzero(np.diff(G["g"])) + 1, len(G["g"])]
    res = {}
    for wname, w in (("binary", Y), ("continuous", P)):
        solved = agree = 0
        statuses = {}
        for gid in gids:
            a, b = starts[gid], starts[gid + 1]
            r, ww = G["r"][a:b], w[G["i"][a:b]]
            mdl = Model(log_output=False)
            x = mdl.binary_var_list(b - a)
            mdl.maximize(mdl.sum((float(r[j]) - lam * float(ww[j])) * x[j] for j in range(b - a)))
            mdl.add_constraint(mdl.sum(x) == K)
            s = mdl.solve()
            st = str(mdl.solve_details.status) if mdl.solve_details else "none"
            statuses[st] = statuses.get(st, 0) + 1
            if s is None:
                mdl.end(); continue
            solved += 1
            cp = {j for j in range(b - a) if x[j].solution_value > 0.5}
            adj = r - lam * ww
            srt = set(np.argsort(-adj, kind="stable")[:K].tolist())
            # identical sets, or equal objective value when ties exist
            agree += int(cp == srt or abs(adj[list(cp)].sum() - adj[list(srt)].sum()) < 1e-9)
            mdl.end()
        res[wname] = {"sampled": int(len(gids)), "solved": solved, "agree_with_sort": agree, "status": statuses}
    return res
