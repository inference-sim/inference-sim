# The decision stage

`substrate/decision/` gives, for an ordered pair of points of one case and one front (XI, XT, IT,
XIT), the probability that the first dominates the second (design §15.3). `blis front` reads it.

## The artifact

`artifact/` is the stage of this release: `artifact.json` and, for each front, two XGBoost 2.1 JSON
models. It is embedded in the build (`LoadShipped`); `blis front --artifact DIR` reads another
directory under the same checks (`LoadStage`), so a candidate can be judged before it lands.

| File | What it is |
|---|---|
| `model-<front>.json` | the simulator model: it reads the simulator's predicted differences of the two points, and is monotone in them. It is fitted on pairs whose two points the simulator answered. |
| `fallback-<front>.json` | the fallback model: the same inputs with every simulator difference missing. It is fitted on every pair, answered or not. |
| `artifact.json` | schema version 2 (below), the dominance tolerance, the content hash of the training table, the role counts, the writer's commit, the feature list, and the settings chosen. |

### The combined rule

For a pair whose two points the simulator answers on the front, the simulator model gives the
probabilities. For any other pair the fallback gives them, and every answer for a point the
simulator did not answer is labelled `estimated` (DEC-3). In both cases the model's two estimates for
the pair are normalised together, `pi_pq = g_pq / max(1, g_pq + g_qp)`, so the two probabilities sum to
at most one and do not depend on the order of the pair (DEC-4). A point is on the predicted front unless
some other point dominates it with probability above one half. A point is one the simulator did not answer exactly when the simulator's status for it is not ok, or it gave no value for an objective of the front (the training pipeline's mask, checked against the status column of the simulator's predictions: 1,863 of 2,895 S40 points answered, 1,032 refused or failed, 0 differences). A stage without a fallback (schema
version 1, the test fixture) places only the points the simulator answered.

### Schema version 2

Version 2 adds to version 1 the four `fallback-<front>.json` files and the field
`fallback_masked_features`: the features the fallback never reads (`dX`, `dI`, `dT`, the simulator's
differences). The stage passes them to the fallback as missing, as they were in every row it was fitted on.
`LoadStage` refuses a version 2 artifact with a missing or mismatched fallback model, with no masked list, or
with a masked name that is not a feature. Version 1 artifacts still load.

## The writer

The artifact is written offline, in the way coefficients are (§9), by a writer script kept with the
training-table generator outside this repository. `artifact.json` records the writer's commit and the
content hash of the training table the models were fitted on (DEC-5). The writer:

- fits both models of each front on every training row, with no unit held out for grading;
- drops every row whose ledger unit has role `held` or `anchor`, and every row whose unit's role cannot
  be told from the published lists and the legacy table (DEC-2), and fails when one is among its inputs;
  a record in no ledger unit has no role and is kept (its count is in `chosen.ledger`);
- takes the hyper-parameters each front's cross-fit selected most often (the vote counts are in
  `chosen`);
- gives the models the 29 pair features this package computes (`PairFeatureNames`). The calibration
  stage's columns, the adjusted-TTFT column and the engine-version columns are not inputs.

A new artifact lands only with its held-out decision grades against those of the artifact it replaces and
of the simulator alone (DEC-5, VAL-6).

## Tests

- `combined_test.go`: parity of the combined probabilities, memberships and front against the training
  pipeline's own computation on real points (`testdata/combined/parity.json`, tolerance 1e-6); DEC-3 for
  points the simulator refused; DEC-4 for the combined rule; the loader's refusals.
- `stage_test.go`, `artifact_test.go`, `xgboost_test.go`: the simulator model alone, the evaluator and
  schema version 1, on `testdata/`.
