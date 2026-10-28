"""Exports everything the static web app needs into web/ :
   model.bin / model.json (XGBoost trees + preprocessing constants), districts.json (test districts), results.json, assets/*.png"""
import json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.config import HUBS, TARGET, CAP_VALUE                      # noqa: E402  (needed so joblib can import src.*)
WEB = ROOT / "web"; (WEB / "assets").mkdir(parents=True, exist_ok=True)
R = json.load(open(ROOT / "results" / "metrics.json"))

# ───────────────────────── model ─────────────────────────
pipe = joblib.load(ROOT / "models" / "best_model.joblib")
ct = pipe.named_steps["prep"].named_steps["columns"]; est = pipe.named_steps["model"]
t = {n: (tr, cols) for n, tr, cols in ct.transformers_}
log_sc, log_cols = t["log"][0].named_steps["scale"], list(t["log"][1])
num_sc, num_cols = t["num"][0], list(t["num"][1])
cats = [str(c) for c in t["cat"][0].categories_[0]]
feature_names = list(ct.get_feature_names_out())
booster = est.get_booster(); cfg = json.loads(booster.save_config())
base_score = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))

feat, left, right, val, roots = [], [], [], [], []
for tree_json in booster.get_dump(dump_format="json"):
    tree = json.loads(tree_json); nodes = {}
    def walk(n):
        nodes[n["nodeid"]] = n
        for c in n.get("children", []): walk(c)
    walk(tree)
    base = len(feat); ids = sorted(nodes); local = {nid: base + i for i, nid in enumerate(ids)}; roots.append(local[0])
    for nid in ids:
        n = nodes[nid]
        if "leaf" in n:
            feat.append(-1); left.append(0); right.append(0); val.append(n["leaf"])
        else:
            feat.append(int(n["split"][1:])); left.append(local[n["yes"]]); right.append(local[n["no"]]); val.append(n["split_condition"])
N = len(feat)
with open(WEB / "model.bin", "wb") as f:                           # little-endian typed arrays, 4-byte aligned first
    f.write(np.array(left, "<i4").tobytes()); f.write(np.array(right, "<i4").tobytes())
    f.write(np.array(val, "<f4").tobytes()); f.write(np.array(feat, "<i2").tobytes())
train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
raw_cols = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population", "households", "median_income"]
meta = {"nodes": N, "roots": roots, "base_score": base_score, "features": feature_names, "log_cols": log_cols, "log_mean": log_sc.mean_.tolist(), "log_scale": log_sc.scale_.tolist(),
        "num_cols": num_cols, "num_mean": num_sc.mean_.tolist(), "num_scale": num_sc.scale_.tolist(), "categories": cats,
        "hubs": {k: list(v) for k, v in HUBS.items()}, "train_range": {c: [float(train[c].min()), float(train[c].max())] for c in raw_cols}}
json.dump(meta, open(WEB / "model.json", "w"), separators=(",", ":"))

# ───────────────────────── districts (test set: never seen in training) ─────────────────────────
test = pd.read_csv(ROOT / "data" / "processed" / "test.csv"); pred = pd.read_csv(ROOT / "results" / "test_predictions.csv")
assert len(test) == len(pred) and np.allclose(test[TARGET].values, pred["actual"].values)
d = {"lon": test.longitude.round(2).tolist(), "lat": test.latitude.round(2).tolist(), "actual": test[TARGET].round().astype(int).tolist(),
     "pred": pred["predicted"].round().astype(int).tolist(), "income": test.median_income.round(2).tolist(), "ocean": [cats.index(c) for c in test.ocean_proximity]}
json.dump({"categories": cats, **d}, open(WEB / "districts.json", "w"), separators=(",", ":"))

# ───────────────────────── results used by the page ─────────────────────────
o1, o2, o3, o4, o5, o6, q = (R[k] for k in ("o1", "o2", "o3", "o4", "o5", "o6", "quality")); te, cv = o5["test_table"], o5["cv_table"]; best = o5["best_model"]
order = o5["rank_by_cv"]; sc, rc = o3["spatial_contrasts"], o4["ratio_contrasts"]; cen = o6["censoring"]; me = o1["masked_experiment"]
usd = lambda x: f"${x:,.0f}"
findings = [
 {"q": "Can the 207 missing bedroom counts be filled in safely?", "a": f"Yes. Missingness shows no relation to other variables, and a k-NN imputer recovers hidden values with an error of {me['knn']['RMSE']:.0f} rooms against {me['median']['RMSE']:.0f} for the median. The effect on price prediction is negligible."},
 {"q": "How strongly does income drive house value?", "a": f"Strongly: correlation {o2['pearson']:.2f}. Every extra $10,000 of district income goes with about {usd(o2['ols_slope'])} more in house value, though income alone explains only {o2['ols_r2']*100:.0f} % of the variation."},
 {"q": "Does location matter beyond income?", "a": f"A great deal. Ocean proximity has a large effect and prices are strongly clustered in space (Moran's I = {o3['moran_I']['I']:.2f}). Adding distances to five major cities cut error by {abs(sc[0]['pct_change']):.1f} % for a linear model and {abs(sc[1]['pct_change']):.1f} % for gradient boosting."},
 {"q": "Do per-household ratios help?", "a": f"Yes, modestly and consistently: rooms, bedrooms and people per household cut error by {abs(rc[0]['pct_change']):.1f} % to {abs(rc[1]['pct_change']):.1f} %, and every comparison was statistically significant."},
 {"q": "Which model predicts best?", "a": f"{best}, with a test error (RMSE) of {usd(te[best]['RMSE'])} and R² of {te[best]['R2']:.2f}. It beats the runner-up, {o5['runner_up']}, by a small but statistically reliable margin."},
 {"q": "Where does it go wrong?", "a": f"On the most expensive districts. Values are capped at $500,001 in the data; those districts are {cen['pct_test_capped']:.1f} % of the test set but cause {cen['rmse_share_from_capped_rows_pct']:.0f} % of the squared error."},
]
res = {"best": best, "models": [{"name": m, "cv_rmse": cv[m]["CV_RMSE"], "rmse": te[m]["RMSE"], "mae": te[m]["MAE"], "r2": te[m]["R2"]} for m in order],
       "medape": te[best]["MedAPE_%"], "within20": te[best]["Within20%"], "rmse": te[best]["RMSE"], "mae": te[best]["MAE"], "r2": te[best]["R2"], "cap": CAP_VALUE,
       "n_test": R["split"]["n_test"], "n_train": R["split"]["n_train"], "baseline_rmse": te["Dummy (mean)"]["RMSE"], "findings": findings}
json.dump(res, open(WEB / "results.json", "w"), indent=1)

# ───────────────────────── figures ─────────────────────────
for name in ["fig04_geography", "fig06_income_relationship", "fig08_ocean_proximity", "fig12_model_comparison", "fig13_pred_vs_actual", "fig17_residual_map", "fig19_shap_summary", "fig20_partial_dependence"]:
    im = Image.open(ROOT / "figures" / f"{name}.png").convert("RGB"); w = min(1500, im.width); im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    im.save(WEB / "assets" / f"{name}.png", optimize=True)
print(f"exported: {N:,} nodes, {len(roots)} trees; model.bin {(WEB/'model.bin').stat().st_size/1e6:.2f} MB; districts {len(test):,}")
