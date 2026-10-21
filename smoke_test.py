import sys, numpy as np, pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

R = Path(sys.argv[1])           # run folder
STEP = 10                       # use every 10th second (rows are 1 s apart and near-identical)
HORIZONS = [15, 30, 45, 60]
GAP = pd.Timedelta(minutes=60)  # purge gap = largest horizon

feat = pd.read_csv(R / "causal_features.csv")
if "timestamp" not in feat.columns:
    feat = feat.rename(columns={feat.columns[0]: "timestamp"})
lab = pd.read_csv(R / "forecast_labels.csv")
for d in (feat, lab):
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)

lab_cols = [c for c in lab.columns if c.startswith(("y_", "valid_"))]
feat = feat.drop(columns=[c for c in lab_cols + ["segment_id"] if c in feat.columns])
X_cols = feat.select_dtypes("number").columns.tolist()
df = feat.merge(lab, on="timestamp", how="inner").iloc[::STEP].reset_index(drop=True)
df[X_cols] = df[X_cols].replace([np.inf, -np.inf], np.nan)
print(f"{len(df)} rows, {len(X_cols)} features, {df.timestamp.min()} -> {df.timestamp.max()}")

cat = pd.read_csv(R / "reference_event_catalogue_validated.csv")
onsets = pd.to_datetime(cat["onset_utc"], utc=True)

t0, t1 = df.timestamp.min(), df.timestamp.max()
a, b = t0 + 0.6 * (t1 - t0), t0 + 0.8 * (t1 - t0)
parts = {
    "train": df.timestamp < a - GAP,
    "val":   (df.timestamp >= a) & (df.timestamp < b - GAP),
    "test":  df.timestamp >= b,
}
rng = {"train": (t0, a - GAP), "val": (a, b - GAP), "test": (b, t1)}
for k, (s, e) in rng.items():
    print(f"{k}: {((onsets >= s) & (onsets < e)).sum()} flare onsets")

def scores(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    tss = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = 2 * (tp * tn - fp * fn) / den if den else float("nan")
    prec = tp / max(tp + fp, 1)
    return dict(TSS=tss, HSS=hss, precision=prec, recall=tp / max(tp + fn, 1), TP=tp, FP=fp, FN=fn, TN=tn)

for h in HORIZONS:
    print(f"\n=== horizon {h} min ===")
    ok = df[f"valid_{h}m"].fillna(False).astype(bool)
    y = df[f"y_{h}m"].fillna(0).astype(int)
    d = {k: (m & ok) for k, m in parts.items()}
    n_pos = {k: int(y[m].sum()) for k, m in d.items()}
    print("positive rows (train/val/test):", n_pos["train"], n_pos["val"], n_pos["test"])
    if n_pos["train"] == 0 or y[d["train"]].nunique() < 2:
        print("SKIP: no positives in train"); continue
    ytr = y[d["train"]]
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.5, scale_pos_weight=(ytr == 0).sum() / max(ytr.sum(), 1),
                      eval_metric="aucpr", n_jobs=-1, random_state=42)
    m.fit(df.loc[d["train"], X_cols], ytr)
    thr = 0.5
    if n_pos["val"] > 0:
        pv = m.predict_proba(df.loc[d["val"], X_cols])[:, 1]
        yv = y[d["val"]]
        thr = max(np.linspace(0.05, 0.95, 91), key=lambda t: scores(yv, pv >= t)["TSS"])
    else:
        print("WARNING: no positives in validation, using threshold 0.5")
    if n_pos["test"] == 0:
        print("Test has no positives: metrics undefined"); continue
    pt = m.predict_proba(df.loc[d["test"], X_cols])[:, 1]
    yt = y[d["test"]]
    r = scores(yt, pt >= thr)
    r["AUC"] = roc_auc_score(yt, pt)
    print(f"threshold={thr:.2f}  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))