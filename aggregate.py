"""Aggregate per-seed results into a compact summary (means, SDs, paired tests)."""
import os, sys, json, gzip, base64
from collections import defaultdict
import numpy as np
from scipy import stats as st

KEYS = ["P@K", "R@K", "NDCG@K", "HR@K", "MRR@K", "AD", "CC", "LTC", "ARP_clicks", "ARP_pct",
        "APLT", "PR", "Gini", "ED", "NDCE", "CTS", "imp_AUC", "n_lists", "catalogue", "tail_catalogue"]
TEST_KEYS = ["P@K", "R@K", "NDCG@K", "HR@K", "MRR@K", "CC", "LTC", "ARP_pct", "APLT", "Gini", "ED", "NDCE", "CTS"]


def load(out):
    d = {}
    for f in os.listdir(out):
        if f.endswith(".jsonl"):
            d[f[:-6]] = [json.loads(l) for l in open(os.path.join(out, f)) if l.strip()]
    return d


def summarise(out):
    d = load(out)
    grid = d.get("grid", [])
    cell = defaultdict(dict)  # (ds, model, mode, w, lam) -> seed -> row
    for r in grid:
        cell[(r["dataset"], r["model"], r["mode"], r["w"], r["lambda"])][r["seed"]] = r
    agg = []
    for key, bys in cell.items():
        ds, model, mode, w, lam = key
        row = {"dataset": ds, "model": model, "mode": mode, "w": w, "lambda": lam, "n_seeds": len(bys)}
        for k in KEYS:
            v = [bys[s][k] for s in bys if k in bys[s]]
            if v:
                row[k] = float(np.mean(v)); row[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        base = cell.get((ds, model, mode, "none", 0.0), {})
        seeds = sorted(set(bys) & set(base))
        if w not in ("none",) and len(seeds) >= 3:
            for k in TEST_KEYS:
                a = np.array([bys[s][k] for s in seeds]); b = np.array([base[s][k] for s in seeds])
                dlt = a - b
                row[k + "_diff"] = float(dlt.mean())
                if np.allclose(dlt, 0):
                    row[k + "_p"] = 1.0
                else:
                    row[k + "_p"] = float(st.ttest_rel(a, b).pvalue)
        agg.append(row)
    runs = []
    for r in d.get("run", []):
        rr = {k: v for k, v in r.items() if k != "history"}
        runs.append(rr)
    hist = [r for r in d.get("run", []) if r["seed"] == 42]
    # architecture comparison on baseline NDCG (impression level), Welch t-test
    comp = []
    for ds in sorted({r["dataset"] for r in grid}):
        vals = {}
        for m in ("MF", "H-NCF", "EH-NCF"):
            c = cell.get((ds, m, "impression", "none", 0.0), {})
            if c:
                vals[m] = c
        ms = list(vals)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                for k in ("NDCG@K", "imp_AUC", "APLT", "Gini", "ARP_pct"):
                    a = [vals[ms[i]][s][k] for s in vals[ms[i]]]; b = [vals[ms[j]][s][k] for s in vals[ms[j]]]
                    if len(a) > 1 and len(b) > 1:
                        comp.append({"dataset": ds, "a": ms[i], "b": ms[j], "metric": k,
                                     "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
                                     "p": float(st.ttest_ind(a, b, equal_var=False).pvalue)})
    return {"stats": d.get("stats", []), "cplex": d.get("cplex", []), "sig_within_seed": d.get("sig_within_seed", []),
            "runs": runs, "hist42": hist, "agg": agg, "arch_tests": comp}


def emit(out, tag):
    s = json.dumps(summarise(out), separators=(",", ":")).encode()
    blob = base64.b64encode(gzip.compress(s, 9)).decode()
    n = (len(blob) + 3499) // 3500
    for k in range(n):
        print(f"SUMMARY {tag} {k + 1}/{n} {blob[k * 3500:(k + 1) * 3500]}", flush=True)
    print(f"SUMMARY_END {tag} {n}", flush=True)


if __name__ == "__main__":
    emit(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "final")
