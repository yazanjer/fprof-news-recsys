"""Create a Zenodo *draft* deposition with the trained weights and results.

Usage: ZENODO_TOKEN=... python scripts/zenodo_deposit.py <file> [<file> ...]
The draft is not published; publish it from the Zenodo web interface after checking it.
Prints the deposition id, the reserved DOI and the edit URL.
"""
import os, sys, json, requests

API = os.environ.get("ZENODO_API", "https://zenodo.org/api")
TOKEN = os.environ["ZENODO_TOKEN"]
P = {"access_token": TOKEN}

META = {"metadata": {
    "title": "F-PROF: trained click models and experiment outputs for fairness-aware "
             "post-ranking optimisation in neural news recommendation (MIND, EB-NeRD)",
    "upload_type": "dataset",
    "description": (
        "<p>Trained weights and complete experiment outputs accompanying the PhD thesis "
        "<em>Fairness-Aware Post-Ranking Optimisation for Neural News Recommendation Systems</em> "
        "(Taylor's University).</p>"
        "<p><strong>Models.</strong> Matrix Factorization (MF), Hybrid NCF (H-NCF) and Enhanced Hybrid NCF "
        "(EH-NCF), TensorFlow/Keras, trained as click predictors on MIND-small and EB-NeRD-small with the "
        "official time-ordered splits; ten seeds (42-51) per model and dataset (60 weight files). Each "
        "dataset folder contains model_config.json (model sizes), item_ids.json and user_ids.json (index "
        "maps) and popularity.npz (training-period click counts, popularity percentile P and binary "
        "popularity Y). Article features are regenerated deterministically from the public datasets by "
        "data.py in the code repository.</p>"
        "<p><strong>Results.</strong> Per-seed training histories, the full re-ranking grid (11 values of "
        "lambda, binary and continuous popularity penalties, impression-level and user-level lists, "
        "accuracy and fairness metrics), the CPLEX verification and the within-seed Wilcoxon tests "
        "(JSON Lines), plus the aggregated summaries used to generate every table and figure of the "
        "thesis.</p>"
        "<p>The datasets themselves are not redistributed; obtain MIND and EB-NeRD from their "
        "providers under their licences.</p>"),
    "creators": [{"name": "Esseid, Kusai Mosbah", "affiliation": "Taylor's University"},
                 {"name": "Aljeroudi, Yazan", "affiliation": "Rachis Systems"}],
    "keywords": ["news recommendation", "popularity bias", "fairness-aware re-ranking",
                 "neural collaborative filtering", "exposure fairness", "MIND", "EB-NeRD",
                 "trained models"],
    "license": "cc-by-4.0",
    "access_right": "open",
    "related_identifiers": [{"identifier": "https://github.com/yazanjer/fprof-news-recsys",
                             "relation": "isSupplementTo", "resource_type": "software"}],
    "prereserve_doi": True,
}}


def main(files):
    r = requests.post(f"{API}/deposit/depositions", params=P, json={})
    r.raise_for_status()
    dep = r.json()
    bucket = dep["links"]["bucket"]
    for f in files:
        with open(f, "rb") as fh:
            u = requests.put(f"{bucket}/{os.path.basename(f)}", data=fh, params=P)
        u.raise_for_status()
        print("UPLOADED", os.path.basename(f), u.json().get("checksum"), flush=True)
    m = requests.put(f"{API}/deposit/depositions/{dep['id']}", params=P,
                     data=json.dumps(META), headers={"Content-Type": "application/json"})
    m.raise_for_status()
    d = m.json()
    doi = d["metadata"].get("prereserve_doi", {}).get("doi")
    print("ZENODO_DRAFT", json.dumps({"id": d["id"], "doi": doi,
                                      "edit": d["links"].get("html"),
                                      "files": [x["filename"] for x in d.get("files", [])]}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
