# F-PROF: Fairness-Aware Post-Ranking Optimisation for Neural News Recommendation

Code for the PhD thesis *Fairness-Aware Post-Ranking Optimisation for Neural News
Recommendation Systems* (Kusai Mosbah Esseid, Taylor's University). The pipeline trains three
click-prediction models on MIND-small and EB-NeRD-small, re-ranks their candidate lists with
F-PROF over a grid of fairness weights, and evaluates accuracy and fairness with significance
tests. Every table and figure of Chapter 4 of the thesis is generated from its outputs by
`make_results.py`.

**Trained models and results:** Zenodo, [10.5281/zenodo.23021433](https://doi.org/10.5281/zenodo.23021433)
(60 weight files, index maps, per-seed results and aggregated summaries).

## Method in brief

- **Scorers** (`models.py`): Matrix Factorization with biases (MF); Hybrid NCF (GMF branch plus a
  title-feature tower and an MLP head); Enhanced Hybrid NCF (adds abstract, category and
  subcategory features). No model contains a fairness term.
- **Data** (`data.py`): official time-ordered splits (MIND: days 1-6 / day 7; EB-NeRD: train week /
  validation week); the last 10% of training impressions for early stopping; four sampled
  non-clicked candidates per click. Popularity is computed from training-period clicks only:
  binary `Y` (above the 75th percentile of clicked items) and percentile `P`.
- **F-PROF** (`evaluate.py`): for each list, select `K` items maximising
  `sum_j (r_j - lambda * w_j) x_j` subject to `sum_j x_j = K`, with `w = Y` or `w = P`. With a
  single cardinality constraint the optimum is the `K` largest adjusted scores; IBM CPLEX is used
  only to verify this on sampled lists.
- **Evaluation**: P, R, NDCG, HR, MRR@5; aggregate diversity, catalogue and long-tail coverage;
  average popularity, APLT, popularity ratio, Gini of exposure; exposure disparity and NDCE;
  composite trade-off score. Impression-level and user-level lists;
  `lambda in {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1}`; ten seeds (42-51);
  paired t-tests against `lambda = 0` with Holm adjustment, Wilcoxon tests within seed.

## Running

```bash
pip install -r requirements.txt
SAVE_MODELS=1 bash scripts/run_all.sh          # downloads the datasets on first use
python make_results.py results <thesis-dir>   # tables -> <thesis-dir>/chapter 4/generated, figures -> plots/generated
```

`run.py` resumes from the last completed (dataset, model, seed). The reported run used one
NVIDIA RTX A4000 with the `tensorflow/tensorflow:2.16.1-gpu` image
(`scripts/runpod_train_and_deposit.sh`).

## Using the trained weights

Each dataset folder of the Zenodo archive contains `model_config.json`, `item_ids.json`,
`user_ids.json`, `popularity.npz` and `<MODEL>_seed<SEED>.weights.h5`.

```python
import json, numpy as np, tensorflow as tf
from models import MODELS
import data as D

cfg = json.load(open("MIND/model_config.json"))
kw = dict(num_users=cfg["n_users"], num_items=cfg["n_items"])
kw_eh = dict(kw, num_categories=cfg["num_categories"], num_subcategories=cfg["num_subcategories"])
m = MODELS["EH-NCF"](**kw_eh)

ds = D.load("MIND")                    # regenerates the article features deterministically
f = ds["feat"]
x = {"user_id": np.array([0]), "news_id": np.array([1]),
     "title_embedding": f["title"][[1]], "abstract_embedding": f["abstract"][[1]],
     "category_id": f["category"][[1]], "subcategory_id": f["subcategory"][[1]]}
m(x)                                   # build the variables
m.load_weights("MIND/EH-NCF_seed42.weights.h5")
```

Item index `i` corresponds to `item_ids[i-1]` (0 is padding); user index `u` corresponds to
`user_ids[u-1]` (0 is the shared embedding for users not seen in training).

## Data

MIND and EB-NeRD are not redistributed here. They are downloaded from their providers by
`data.py` and are subject to their own licences (MIND: Microsoft Research License Terms;
EB-NeRD: Ekstra Bladet terms of use).

## Citation

See `CITATION.cff`. Authors: Kusai Mosbah Esseid, Yazan Aljeroudi.

## Licence

Code: MIT. Trained models and results on Zenodo: CC BY 4.0.
