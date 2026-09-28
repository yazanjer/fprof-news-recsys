"""End-to-end F-PROF benchmark.

For every dataset x model x seed: train the click model (early stopping on
validation AUC), score every test candidate, and evaluate the re-ranking grid.
Results are appended to results/*.jsonl and echoed to stdout as RESULT lines.
"""
import os, sys, json, time, argparse, gzip, base64
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

import data as D
import evaluate as E
from models import MODELS

LAMBDAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
OUT = os.environ.get("FPROF_OUT", "/workspace/results")
SAVE = os.environ.get("FPROF_SAVE_MODELS", "")  # directory for trained weights; empty = do not save


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def emit(kind, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{kind}.jsonl"), "a") as f:
        f.write(json.dumps(obj) + "\n")
    if kind in ("stats", "cplex"):
        print("RESULT", kind, json.dumps(obj), flush=True)


def export_vocab(ds, d):
    """Write the index maps and model sizes needed to reuse the saved weights."""
    os.makedirs(d, exist_ok=True)
    meta = {"n_users": int(ds["n_users"]), "n_items": int(ds["n_items"]),
            "num_categories": int(ds["feat"]["category"].max() + 1),
            "num_subcategories": int(ds["feat"]["subcategory"].max() + 1),
            "title_features": ds["feat"]["title_source"],
            "index_note": "item index = position in item_ids + 1 (0 = padding); "
                          "user index = position in user_ids + 1 (0 = user unseen in training)"}
    json.dump(meta, open(os.path.join(d, "model_config.json"), "w"), indent=1)
    if "item_ids" in ds:
        json.dump(ds["item_ids"], open(os.path.join(d, "item_ids.json"), "w"))
        json.dump(ds["user_ids"], open(os.path.join(d, "user_ids.json"), "w"))
    np.savez_compressed(os.path.join(d, "popularity.npz"), clicks=ds["clicks"], P=ds["P"], Y=ds["Y"])


def make_ds(ds, rows, batch, shuffle, seed):
    f = ds["feat"]
    T = tf.constant(f["title"]); A = tf.constant(f["abstract"])
    C = tf.constant(f["category"]); S = tf.constant(f["subcategory"])
    y = rows.get("y", np.zeros(len(rows["u"]), np.float32))
    d = tf.data.Dataset.from_tensor_slices((rows["u"].astype(np.int64), rows["i"].astype(np.int64), y.astype(np.float32)))
    if shuffle:
        d = d.shuffle(min(len(y), 200_000), seed=seed, reshuffle_each_iteration=True)

    def fmt(u, i, yy):
        return ({"user_id": u, "news_id": i, "title_embedding": tf.gather(T, i),
                 "abstract_embedding": tf.gather(A, i), "category_id": tf.gather(C, i),
                 "subcategory_id": tf.gather(S, i)}, yy)
    return d.batch(batch).map(fmt, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def build(name, ds):
    kw = dict(num_users=ds["n_users"], num_items=ds["n_items"])
    if name == "EH-NCF":
        kw.update(num_categories=int(ds["feat"]["category"].max() + 1),
                  num_subcategories=int(ds["feat"]["subcategory"].max() + 1))
    return MODELS[name](**kw)


def train_one(ds, model_name, seed, args):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    m = build(model_name, ds)
    m.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.AUC(name="AUC")])
    tr = make_ds(ds, ds["train"], args.batch, True, seed)
    va = make_ds(ds, ds["val"], 8192, False, seed)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_AUC", mode="max", patience=args.patience, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    t0 = time.time()
    h = m.fit(tr, validation_data=va, epochs=args.epochs, callbacks=cb, verbose=2)
    hist = {k: [float(x) for x in v] for k, v in h.history.items()}
    best = int(np.argmax(hist["val_AUC"]))
    return m, hist, best, time.time() - t0


def score(m, ds):
    te = make_ds(ds, ds["test"], 16384, False, 0)
    return np.concatenate([m(x, training=False).numpy().ravel() for x, _ in te]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["MIND", "EBNeRD"])
    ap.add_argument("--models", nargs="+", default=["MF", "H-NCF", "EH-NCF"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 52)))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=5)
    args = ap.parse_args()
    log("GPUs:", tf.config.list_physical_devices("GPU"))
    done = set()
    p = os.path.join(OUT, "run.jsonl")
    if os.path.exists(p):
        for line in open(p):
            r = json.loads(line); done.add((r["dataset"], r["model"], r["seed"]))
    for dname in args.datasets:
        ds = D.load(dname)
        emit("stats", ds["stats"])
        if SAVE:
            export_vocab(ds, os.path.join(SAVE, dname))
        for model_name in args.models:
            for seed in args.seeds:
                if (dname, model_name, seed) in done:
                    continue
                m, hist, best, secs = train_one(ds, model_name, seed, args)
                s = score(m, ds)
                glob_auc = float(roc_auc_score(ds["test"]["y"], s))
                keep = seed == args.seeds[0]
                rows, sig = E.rerank_grid(ds["test"], s, ds["clicks"], ds["P"], ds["Y"], LAMBDAS, args.K, keep_ndcg=keep)
                for r in rows:
                    emit("grid", {"dataset": dname, "model": model_name, "seed": seed, **r})
                for r in sig:
                    emit("sig_within_seed", {"dataset": dname, "model": model_name, "seed": seed, **r})
                if keep:
                    try:
                        emit("cplex", {"dataset": dname, "model": model_name, "seed": seed,
                                       **E.cplex_check(ds["test"], s, ds["P"], ds["Y"], 0.5, args.K)})
                    except Exception as e:
                        emit("cplex", {"dataset": dname, "model": model_name, "seed": seed, "error": repr(e)})
                if SAVE:
                    mdir = os.path.join(SAVE, dname)
                    os.makedirs(mdir, exist_ok=True)
                    m.save_weights(os.path.join(mdir, f"{model_name}_seed{seed}.weights.h5"))
                emit("run", {"dataset": dname, "model": model_name, "seed": seed, "best_epoch": best + 1,
                             "epochs_run": len(hist["loss"]), "val_AUC": hist["val_AUC"][best],
                             "val_loss": hist["val_loss"][best], "test_global_AUC": glob_auc,
                             "train_seconds": secs, "history": hist})
                log("done", dname, model_name, seed, "valAUC=%.4f testAUC=%.4f" % (hist["val_AUC"][best], glob_auc))
    # final bundle to stdout so results survive without file transfer
    blob = base64.b64encode(gzip.compress(b"".join(open(os.path.join(OUT, f), "rb").read() + b"\n#FILE " + f.encode() + b"\n"
                                                   for f in sorted(os.listdir(OUT)) if f.endswith(".jsonl")))).decode()
    for k in range(0, len(blob), 4000):
        print("BUNDLE", k // 4000, blob[k:k + 4000], flush=True)
    print("BUNDLE_END", flush=True)


if __name__ == "__main__":
    main()
