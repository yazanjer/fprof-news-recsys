"""Dataset download and preprocessing for MIND-small and EB-NeRD-small.

Both datasets are used with their official, time-ordered splits:
  * MIND-small: MINDsmall_train (days 1-6 of the collection week) for training,
    MINDsmall_dev (day 7) as the held-out test period.
  * EB-NeRD-small: train/ for training, validation/ (the following week) as test.
The last 10% of training impressions (by time) are held out for early stopping.
Popularity statistics are computed from training-period clicks only.
"""
import os, io, zipfile, json, subprocess, pickle, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

ROOT = os.environ.get("FPROF_DATA", "/workspace/data")

MIND_URLS = {
    "train": ["https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
              "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip",
              "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_train.zip"],
    "dev": ["https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
            "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip",
            "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_dev.zip"],
}
EB_URL = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip"
EB_BERT_URL = "https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/google_bert_base_multilingual_cased.zip"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _fetch(urls, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return True
    for u in urls:
        r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", dest, u])
        if r.returncode == 0 and os.path.getsize(dest) > 1000:
            log("downloaded", u)
            return True
        log("failed", u)
    return False


def _unzip(z, d):
    os.makedirs(d, exist_ok=True)
    with zipfile.ZipFile(z) as f:
        f.extractall(d)


def _text_pca(texts, n=64, stop_words=None, seed=42):
    vec = TfidfVectorizer(max_features=10000, stop_words=stop_words, token_pattern=r"(?u)\b\w+\b", min_df=2)
    X = vec.fit_transform(texts)
    k = min(n, X.shape[1] - 1)
    Z = PCA(n_components=k, svd_solver="arpack", random_state=seed).fit_transform(X)
    if k < n:
        Z = np.hstack([Z, np.zeros((Z.shape[0], n - k))])
    return Z.astype(np.float32)


def _dense_pca(M, n=64, seed=42):
    return PCA(n_components=n, random_state=seed).fit_transform(M).astype(np.float32)


# ---------------------------------------------------------------- MIND
def _mind_raw():
    d = os.path.join(ROOT, "mind")
    os.makedirs(d, exist_ok=True)
    for part in ("train", "dev"):
        z = os.path.join(d, f"{part}.zip")
        if not os.path.exists(os.path.join(d, part, "behaviors.tsv")):
            if not _fetch(MIND_URLS[part], z):
                raise RuntimeError("MIND download failed")
            _unzip(z, os.path.join(d, part))
    cols_b = ["impression_id", "user_id", "time", "history", "impressions"]
    cols_n = ["news_id", "category", "subcategory", "title", "abstract", "url", "te", "ae"]
    rb = lambda p: pd.read_csv(os.path.join(d, p, "behaviors.tsv"), sep="\t", names=cols_b)
    rn = lambda p: pd.read_csv(os.path.join(d, p, "news.tsv"), sep="\t", names=cols_n, quoting=3)
    news = pd.concat([rn("train"), rn("dev")]).drop_duplicates("news_id").reset_index(drop=True)
    btr, bte = rb("train"), rb("dev")
    for b in (btr, bte):
        b["time"] = pd.to_datetime(b["time"], format="%m/%d/%Y %I:%M:%S %p")

    def explode(b):
        s = b[["impression_id", "user_id", "time", "impressions"]].copy()
        s["impressions"] = s["impressions"].str.split()
        s = s.explode("impressions")
        p = s["impressions"].str.rsplit("-", n=1, expand=True)
        s["item"] = p[0]
        s["label"] = p[1].astype(np.int8)
        return s.drop(columns="impressions")

    news["title_text"] = news["title"].fillna("")
    news["abs_text"] = news["abstract"].fillna("")
    return explode(btr), explode(bte), news, "english", None


# ---------------------------------------------------------------- EB-NeRD
def _ebnerd_raw():
    d = os.path.join(ROOT, "ebnerd")
    os.makedirs(d, exist_ok=True)
    z = os.path.join(d, "ebnerd_small.zip")
    if not os.path.exists(os.path.join(d, "small", "articles.parquet")):
        if not _fetch([EB_URL], z):
            raise RuntimeError("EB-NeRD download failed")
        _unzip(z, os.path.join(d, "small"))
    bert = None
    zb = os.path.join(d, "bert.zip")
    try:
        if _fetch([EB_BERT_URL], zb):
            _unzip(zb, os.path.join(d, "bert"))
            for dp, _, fs in os.walk(os.path.join(d, "bert")):
                for f in fs:
                    if f.endswith(".parquet"):
                        bert = pd.read_parquet(os.path.join(dp, f))
    except Exception as e:  # recorded, not silent
        log("BERT artefact unavailable:", repr(e))
    base = os.path.join(d, "small")
    news = pd.read_parquet(os.path.join(base, "articles.parquet"))
    news = news.rename(columns={"article_id": "news_id"})
    news["subcategory"] = news["subcategory"].apply(lambda x: int(x[0]) if x is not None and len(x) > 0 else -1)
    news["title_text"] = news["title"].fillna("")
    news["abs_text"] = news["subtitle"].fillna("")

    def explode(split):
        b = pd.read_parquet(os.path.join(base, split, "behaviors.parquet"),
                            columns=["impression_id", "user_id", "impression_time", "article_ids_inview", "article_ids_clicked"])
        b = b.rename(columns={"impression_time": "time"})
        b["clicked"] = b["article_ids_clicked"].apply(lambda x: set(int(v) for v in x))
        s = b[["impression_id", "user_id", "time", "article_ids_inview", "clicked"]].explode("article_ids_inview")
        s = s.dropna(subset=["article_ids_inview"])
        s["item"] = s["article_ids_inview"].astype(np.int64)
        s["label"] = [int(i in c) for i, c in zip(s["item"], s["clicked"])]
        s["label"] = s["label"].astype(np.int8)
        return s[["impression_id", "user_id", "time", "item", "label"]]

    if bert is not None:
        bert = bert.rename(columns={"article_id": "news_id"})
        bert = bert.rename(columns={bert.columns[1]: "vec"})
    return explode("train"), explode("validation"), news, None, bert


# ---------------------------------------------------------------- common
def load(name, neg_ratio=4, val_frac=0.10, seed=42):
    cache = os.path.join(ROOT, f"{name}_prepared.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    log("preparing", name)
    tr, te, news, stop, bert = (_mind_raw() if name == "MIND" else _ebnerd_raw())

    # item vocabulary: all articles in the catalogue (index 0 = padding)
    news = news.drop_duplicates("news_id").reset_index(drop=True)
    item_index = {k: j + 1 for j, k in enumerate(news["news_id"].tolist())}
    n_items = len(news) + 1
    tr = tr[tr["item"].isin(item_index)].copy()
    te = te[te["item"].isin(item_index)].copy()
    tr["i"] = tr["item"].map(item_index).astype(np.int64)
    te["i"] = te["item"].map(item_index).astype(np.int64)

    # users: vocabulary from the training period; unseen users -> 0
    users = pd.Index(tr["user_id"].unique())
    uidx = {k: j + 1 for j, k in enumerate(users)}
    n_users = len(users) + 1
    tr["u"] = tr["user_id"].map(uidx).astype(np.int64)
    te["u"] = te["user_id"].map(uidx).fillna(0).astype(np.int64)

    # content features
    feat = {}
    if bert is not None:
        M = np.zeros((n_items, len(bert["vec"].iloc[0])), np.float32)
        for k, v in zip(bert["news_id"], bert["vec"]):
            if k in item_index:
                M[item_index[k]] = np.asarray(v, np.float32)
        feat["title"] = _dense_pca(M, 64)
        feat["title_source"] = "mBERT-base (EB-NeRD artefact) -> PCA64"
    else:
        feat["title"] = np.vstack([np.zeros((1, 64), np.float32), _text_pca(news["title_text"].tolist(), 64, stop)])
        feat["title_source"] = "title TF-IDF -> PCA64"
    feat["abstract"] = np.vstack([np.zeros((1, 64), np.float32), _text_pca(news["abs_text"].tolist(), 64, stop)])
    cat = pd.factorize(news["category"].astype(str))[0] + 1
    sub = pd.factorize(news["subcategory"].astype(str))[0] + 1
    feat["category"] = np.concatenate([[0], cat]).astype(np.int64)
    feat["subcategory"] = np.concatenate([[0], sub]).astype(np.int64)

    # popularity from training-period clicks only
    clicks = np.bincount(tr.loc[tr["label"] == 1, "i"].values, minlength=n_items).astype(np.float64)
    clicked = clicks > 0
    P = np.zeros(n_items)
    P[clicked] = rankdata(clicks[clicked], method="average") / clicked.sum()
    thr = float(np.quantile(clicks[clicked], 0.75))
    Y = (clicks > thr).astype(np.float64)

    # early-stopping hold-out: last val_frac of training impressions by time
    imp_t = tr.groupby("impression_id")["time"].first().sort_values()
    cut = imp_t.index[int(len(imp_t) * (1 - val_frac)):]
    is_val = tr["impression_id"].isin(set(cut))
    rng = np.random.default_rng(seed)

    def sample(df):
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        n = min(len(neg), neg_ratio * len(pos))
        neg = neg.iloc[rng.choice(len(neg), n, replace=False)]
        s = pd.concat([pos, neg])
        return {"u": s["u"].values, "i": s["i"].values, "y": s["label"].values.astype(np.float32)}

    train_rows, val_rows = sample(tr[~is_val]), sample(tr[is_val])
    te = te.sort_values(["impression_id"]).reset_index(drop=True)
    test = {"imp": pd.factorize(te["impression_id"])[0].astype(np.int64), "u": te["u"].values,
            "i": te["i"].values, "y": te["label"].values.astype(np.int8),
            "user_key": pd.factorize(te["user_id"])[0].astype(np.int64)}

    cand = te.groupby("impression_id").size()
    stats = {
        "dataset": name, "n_items_catalogue": int(n_items - 1),
        "train_impressions": int(tr["impression_id"].nunique()),
        "test_impressions": int(te["impression_id"].nunique()),
        "train_users": int(n_users - 1), "test_users": int(te["user_id"].nunique()),
        "test_users_unseen_in_train": int((te.groupby("user_id")["u"].first() == 0).sum()),
        "train_clicks": int(tr["label"].sum()), "test_clicks": int(te["label"].sum()),
        "train_rows_sampled": int(len(train_rows["y"])), "val_rows_sampled": int(len(val_rows["y"])),
        "test_candidate_rows": int(len(te)),
        "candidates_per_test_impression_mean": float(cand.mean()),
        "candidates_per_test_impression_median": float(cand.median()),
        "test_items": int(te["i"].nunique()),
        "test_items_unseen_in_train": int((clicks[te["i"].unique()] == 0).sum()),
        "items_clicked_in_train": int(clicked.sum()),
        "popular_threshold_clicks": thr, "n_popular_items": int(Y.sum()),
        "title_features": feat["title_source"],
        "test_period": [str(te["time"].min()), str(te["time"].max())],
        "train_period": [str(tr["time"].min()), str(tr["time"].max())],
    }
    out = dict(name=name, n_users=n_users, n_items=n_items, feat=feat, clicks=clicks, P=P, Y=Y,
               train=train_rows, val=val_rows, test=test, stats=stats,
               item_ids=news["news_id"].tolist(), user_ids=[str(x) for x in users])
    with open(cache, "wb") as f:
        pickle.dump(out, f, protocol=4)
    log("prepared", name, json.dumps(stats))
    return out
