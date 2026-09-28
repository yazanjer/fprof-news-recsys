"""Build every Chapter 4 / Appendix A table and figure from the aggregated summaries.

Usage: python make_results.py <summary_dir> <thesis_dir>
<summary_dir> holds MIND.json and EBNeRD.json written by aggregate.summarise().
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUM, TH = sys.argv[1], sys.argv[2]
TAB = os.path.join(TH, "chapter 4", "generated"); FIG = os.path.join(TH, "plots", "generated")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
DS = [("MIND", "MIND"), ("EBNeRD", "EB-NeRD")]
MODELS = ["MF", "H-NCF", "EH-NCF"]
LAMS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
COL = {"MF": "#2a78d6", "H-NCF": "#eb6834", "EH-NCF": "#1baf7a"}
MARK = {"MF": "o", "H-NCF": "s", "EH-NCF": "^"}
S = {d: json.load(open(os.path.join(SUM, f"{d}.json"))) for d, _ in DS}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.2,
                     "lines.markersize": 4, "savefig.dpi": 300})


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * ps[i])); adj[i] = run
    return adj


def A(d, model, mode, w, lam):
    for r in S[d]["agg"]:
        if r["model"] == model and r["mode"] == mode and r["w"] == w and abs(r["lambda"] - lam) < 1e-9:
            return r
    return None


def base(d, model, mode):
    return A(d, model, mode, "none", 0.0)


def f(x, n=4):
    return f"{x:.{n}f}"


def pm(r, k, n=4):
    return f"{r[k]:.{n}f}$\\pm${r[k + '_sd']:.{n}f}"


def star(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def pfmt(p):
    if p < 1e-300:
        return "$<10^{-300}$"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{p:.3f}"


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


# Holm adjustment per family: dataset x model x mode x encoding, over lambdas x tested metrics
TEST = ["P@K", "R@K", "NDCG@K", "HR@K", "MRR@K", "CC", "LTC", "ARP_pct", "APLT", "Gini", "ED", "NDCE", "CTS"]
for d, _ in DS:
    fam = {}
    for r in S[d]["agg"]:
        if r["w"] in ("binary", "continuous") and "NDCG@K_p" in r:
            fam.setdefault((r["model"], r["mode"], r["w"]), []).append(r)
    for rows in fam.values():
        ps = [r[k + "_p"] for r in rows for k in TEST]
        adj = holm(ps); i = 0
        for r in rows:
            for k in TEST:
                r[k + "_padj"] = float(adj[i]); i += 1

# ------------------------------------------------------------------ dataset statistics
stats = {}
for d, _ in DS:
    s = S[d]["stats"][0]
    s["cand_mean"] = f"{s['candidates_per_test_impression_mean']:.1f}"
    s["cand_median"] = f"{s['candidates_per_test_impression_median']:.0f}"
    s["popular_threshold_clicks"] = f"{s['popular_threshold_clicks']:.0f}"
    for k, v in s.items():
        if isinstance(v, (int, np.integer)):
            s[k] = f"{v:,}".replace(",", "{,}")
    stats[d] = s
json.dump(stats, open(os.path.join(TAB, "stats.json"), "w"), indent=1)

# ------------------------------------------------------------------ click-model table
lines = []
for d, dn in DS:
    runs = S[d]["runs"]
    for m in MODELS:
        rr = [r for r in runs if r["model"] == m]
        b = base(d, m, "impression")
        va = np.array([r["val_AUC"] for r in rr]); ta = np.array([r["test_global_AUC"] for r in rr])
        ep = np.array([r["best_epoch"] for r in rr])
        lines.append(f"{dn if m == 'MF' else ''} & {m} & {len(rr)} & {va.mean():.4f}$\\pm${va.std(ddof=1):.4f} & "
                     f"{ta.mean():.4f}$\\pm${ta.std(ddof=1):.4f} & {pm(b, 'imp_AUC')} & {ep.mean():.1f} \\\\")
    lines.append("\\midrule" if d == "MIND" else "")
open(os.path.join(TAB, "click_models.tex"), "w").write("\n".join(l for l in lines if l))

# ------------------------------------------------------------------ baseline table (impression, lambda=0)
for d, dn in DS:
    L = []
    for m in MODELS:
        b = base(d, m, "impression")
        L.append(f"{m} & {pm(b,'P@K')} & {pm(b,'R@K')} & {pm(b,'NDCG@K')} & {pm(b,'HR@K')} & {pm(b,'MRR@K')} \\\\")
    open(os.path.join(TAB, f"baseline_acc_{d}.tex"), "w").write("\n".join(L))
    L = []
    for m in MODELS:
        b = base(d, m, "impression")
        L.append(f"{m} & {b['AD']:.0f} & {pm(b,'CC',3)} & {pm(b,'LTC',3)} & {pm(b,'ARP_pct',3)} & {pm(b,'APLT',3)} & "
                 f"{b['PR']:.2f} & {pm(b,'Gini',3)} & {pm(b,'NDCE',3)} \\\\")
    open(os.path.join(TAB, f"baseline_fair_{d}.tex"), "w").write("\n".join(L))

# architecture comparisons
L = []
for d, dn in DS:
    for t in S[d]["arch_tests"]:
        if t["metric"] in ("NDCG@K", "imp_AUC", "APLT"):
            L.append(f"{dn} & {t['metric'].replace('@K','@5').replace('imp_AUC','Impression AUC')} & {t['a']} vs {t['b']} & "
                     f"{t['mean_a']:.4f} & {t['mean_b']:.4f} & {pfmt(t['p'])} \\\\")
open(os.path.join(TAB, "arch_tests.tex"), "w").write("\n".join(L))

# ------------------------------------------------------------------ lambda sweep table (impression level)
KEYS = [("NDCG@K", "NDCG@5", 4), ("P@K", "P@5", 4), ("HR@K", "HR@5", 4), ("APLT", "APLT", 3), ("LTC", "LTC", 3),
        ("ARP_pct", "ARP$_P$", 3), ("Gini", "Gini", 3), ("ED", "ED", 2), ("CTS", "CTS", 3)]
SC = {"ED": 1000.0}
for d, dn in DS:
    for m in MODELS:
        L = []
        b = base(d, m, "impression")
        L.append("0 (baseline) & -- & " + " & ".join(f(b[k] * SC.get(k, 1), n) for k, _, n in KEYS) + " \\\\")
        for w, wn in (("binary", "Binary"), ("continuous", "Continuous")):
            L.append("\\midrule")
            for lam in LAMS[1:]:
                r = A(d, m, "impression", w, lam)
                cells = [f(r[k] * SC.get(k, 1), n) + star(r.get(k + "_padj", 1)) for k, _, n in KEYS]
                L.append(f"{lam:g} & {wn} & " + " & ".join(cells) + " \\\\")
        open(os.path.join(TAB, f"sweep_{d}_{m}.tex"), "w").write("\n".join(L))

# ------------------------------------------------------------------ user-level vs impression-level
L = []
for d, dn in DS:
    for m in MODELS:
        for mode in ("impression", "user"):
            b = base(d, m, mode); r = A(d, m, mode, "binary", 0.2)
            L.append(f"{dn if (m == 'MF' and mode == 'impression') else ''} & {m if mode == 'impression' else ''} & {mode.capitalize()} & "
                     f"{f(b['P@K'])} & {f(b['R@K'])} & {f(b['NDCG@K'])} & {f(b['APLT'],3)} & {f(b['Gini'],3)} & "
                     f"{f(r['NDCG@K'])}{star(r.get('NDCG@K_padj',1))} & {f(r['APLT'],3)}{star(r.get('APLT_padj',1))} \\\\")
    if d == "MIND":
        L.append("\\midrule")
open(os.path.join(TAB, "user_vs_imp.tex"), "w").write("\n".join(L))

# ------------------------------------------------------------------ reversed-sign objective
L = []
for d, dn in DS:
    for m in MODELS:
        b = base(d, m, "impression"); lb = A(d, m, "impression", "legacy_binary", 0.5); cb = A(d, m, "impression", "binary", 0.5)
        L.append(f"{dn if m == 'MF' else ''} & {m} & {f(b['NDCG@K'])} & {f(b['APLT'],3)} & {f(b['ARP_pct'],3)} & "
                 f"{f(lb['NDCG@K'])} & {f(lb['APLT'],3)} & {f(lb['ARP_pct'],3)} & {f(cb['NDCG@K'])} & {f(cb['APLT'],3)} & {f(cb['ARP_pct'],3)} \\\\")
    if d == "MIND":
        L.append("\\midrule")
open(os.path.join(TAB, "legacy.tex"), "w").write("\n".join(L))

# ------------------------------------------------------------------ CPLEX check
L = []
for d, dn in DS:
    for c in S[d]["cplex"]:
        if "error" in c:
            L.append(f"{dn} & {c['model']} & \\multicolumn{{4}}{{l}}{{error: {c['error'][:40]}}} \\\\"); continue
        for w in ("binary", "continuous"):
            x = c[w]
            L.append(f"{dn} & {c['model']} & {w} & {x['sampled']} & {x['solved']} & {x['agree_with_sort']} \\\\")
open(os.path.join(TAB, "cplex.tex"), "w").write("\n".join(L))

# ------------------------------------------------------------------ within-seed Wilcoxon (EH-NCF, impression)
L = []
for d, dn in DS:
    for r in S[d]["sig_within_seed"]:
        if r["model"] == "EH-NCF" and r["mode"] == "impression" and r["lambda"] in (0.01, 0.05, 0.2, 1.0):
            L.append(f"{dn} & {r['w']} & {r['lambda']:g} & {100*r['share_lists_changed']:.1f} & {r['mean_diff']:+.4f} & {pfmt(r['wilcoxon_p'])} \\\\")
open(os.path.join(TAB, "wilcoxon.tex"), "w").write("\n".join(L))

# ------------------------------------------------------------------ seed-level appendix table
L = []
for d, dn in DS:
    for r in sorted(S[d]["runs"], key=lambda r: (MODELS.index(r["model"]), r["seed"])):
        L.append(f"{dn} & {r['model']} & {r['seed']} & {r['val_AUC']:.4f} & {r['test_global_AUC']:.4f} & {r['best_epoch']} & {r['train_seconds']/60:.1f} \\\\")
open(os.path.join(TAB, "seed_level.tex"), "w").write("\n".join(L))

# ------------------------------------------------------------------ figures
# (1) learning curves, seed 42
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.4))
for j, (d, dn) in enumerate(DS):
    for h in S[d]["hist42"]:
        m = h["model"]; H = h["history"]; ep = np.arange(1, len(H["loss"]) + 1)
        ax[0, j].plot(ep, H["loss"], color=COL[m], ls="--", lw=1)
        ax[0, j].plot(ep, H["val_loss"], color=COL[m], marker=MARK[m], label=m)
        ax[1, j].plot(ep, H["AUC"], color=COL[m], ls="--", lw=1)
        ax[1, j].plot(ep, H["val_AUC"], color=COL[m], marker=MARK[m], label=m)
    ax[0, j].set_title(f"{dn}: loss (dashed: training)"); ax[1, j].set_title(f"{dn}: AUC (dashed: training)")
    ax[1, j].set_xlabel("Epoch"); ax[0, j].set_ylabel("Binary cross-entropy"); ax[1, j].set_ylabel("AUC")
    for a in ax[:, j]:
        a.grid(axis="y", lw=0.3, alpha=0.5)
ax[0, 0].legend(frameon=False)
fig.tight_layout(); save(fig, "learning_curves")

# (2) trade-off frontiers: NDCG@5 vs APLT over lambda (impression level)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8))
for j, (d, dn) in enumerate(DS):
    for m in MODELS:
        for w, ls in (("binary", "-"), ("continuous", ":")):
            pts = [base(d, m, "impression")] + [A(d, m, "impression", w, l) for l in LAMS[1:]]
            x = [p["APLT"] for p in pts]; y = [p["NDCG@K"] for p in pts]
            ax[j].plot(x, y, ls=ls, color=COL[m], marker=MARK[m], mfc="white" if w == "continuous" else COL[m],
                       label=f"{m}, {w}")
    ax[j].set_title(dn); ax[j].set_xlabel("APLT (share of long-tail items in top-5)"); ax[j].set_ylabel("NDCG@5")
    ax[j].grid(lw=0.3, alpha=0.5)
ax[1].legend(frameon=False, fontsize=6, loc="lower left")
fig.tight_layout(); save(fig, "tradeoff_ndcg_aplt")

# (3) metrics vs lambda for EH-NCF
MET = [("NDCG@K", "NDCG@5"), ("APLT", "APLT"), ("Gini", "Gini of exposure"), ("ARP_pct", "ARP (percentile)")]
fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.6))
for i, (d, dn) in enumerate(DS):
    for j, (k, kn) in enumerate(MET):
        for m in MODELS:
            for w, ls in (("binary", "-"), ("continuous", ":")):
                pts = [base(d, m, "impression")] + [A(d, m, "impression", w, l) for l in LAMS[1:]]
                y = np.array([p[k] for p in pts]); sd = np.array([p[k + "_sd"] for p in pts])
                ax[i, j].plot(LAMS, y, ls=ls, color=COL[m], marker=MARK[m], ms=3,
                              mfc="white" if w == "continuous" else COL[m], label=f"{m}, {w}")
                ax[i, j].fill_between(LAMS, y - sd, y + sd, color=COL[m], alpha=0.12, lw=0)
        ax[i, j].set_title(f"{dn}: {kn}"); ax[i, j].grid(lw=0.3, alpha=0.5)
        ax[i, j].set_xscale("symlog", linthresh=0.005, linscale=0.5)
        ax[i, j].set_xticks([0, 0.01, 0.1, 1]); ax[i, j].set_xticklabels(["0", "0.01", "0.1", "1"])
        if i == 1:
            ax[i, j].set_xlabel(r"$\lambda$")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "metrics_vs_lambda")

# (4) exposure Lorenz-style summary: Gini vs catalogue coverage
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for j, (d, dn) in enumerate(DS):
    for m in MODELS:
        pts = [base(d, m, "impression")] + [A(d, m, "impression", "binary", l) for l in LAMS[1:]]
        ax[j].plot([p["CC"] for p in pts], [p["Gini"] for p in pts], color=COL[m], marker=MARK[m], label=m)
    ax[j].set_title(f"{dn} (binary penalty)"); ax[j].set_xlabel("Catalogue coverage"); ax[j].set_ylabel("Gini of exposure")
    ax[j].grid(lw=0.3, alpha=0.5)
ax[0].legend(frameon=False)
fig.tight_layout(); save(fig, "coverage_gini")
print("ok")
