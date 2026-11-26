"""Writes the parity fixture of the XGBoost evaluator: model.json and cases.json.

Run:  python3 -I gen.py   (needs xgboost 2.1)

The model is a binary:logistic gbtree fitted on synthetic data (xgboost 2.1.4) with a
monotone constraint, missing values and a base_score that is not one half. cases.json holds
inputs (null is a missing value) and the probabilities Python XGBoost predicts for them.
"""
import json
import os

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
n, d = 4000, 6
X = rng.normal(size=(n, d)).astype(np.float32)
logit = 1.6 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * np.sin(2 * X[:, 2]) + 0.5 * (X[:, 3] > 0.3) - 0.4
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
X[rng.uniform(size=X.shape) < 0.12] = np.nan
X[:, 4] = np.where(rng.uniform(size=n) < 0.5, np.nan, X[:, 4])  # a column that is often missing
y = np.where(np.isnan(X[:, 0]) & (rng.uniform(size=n) < 0.5), 1, y)  # missingness carries signal

clf = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", max_bin=64, learning_rate=0.08,
                        min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, n_jobs=1,
                        random_state=0, monotone_constraints="(1,-1,0,0,0,0)", verbosity=0,
                        max_depth=3, n_estimators=60, base_score=0.35)
clf.fit(X, y)
clf.save_model(os.path.join(HERE, "model.json"))

T = rng.normal(size=(300, d)).astype(np.float32) * 1.5
T[rng.uniform(size=T.shape) < 0.2] = np.nan
T[0] = np.nan                                   # every input missing
T[1] = 0.0                                      # exactly zero
T[2] = X[0]                                     # a training row
p = clf.predict_proba(T)[:, 1]
rows = [[None if np.isnan(v) else float(v) for v in r] for r in T]
json.dump({"xgboost": xgb.__version__, "inputs": rows, "proba": [float(v) for v in p]},
          open(os.path.join(HERE, "cases.json"), "w"))
print("xgboost", xgb.__version__, "cases", len(rows), "nan rows", int(np.isnan(T).any(1).sum()))
