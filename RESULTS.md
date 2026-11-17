# Fixed experiment, November 17, 2026

Question: which acquisition strategy improves held-out classification at an equal
annotation budget? Two synthetic tasks, seeds 17/42/2026 and four strategies give
24 runs, each evaluated at budgets 8/16/24/32/40. Budgets include 8 initial labels.

The same seed/regime uses identical initial/pool/test identities across all
strategies. Features are original uniform draws in [-1,1]²; labels follow a linear
or sinusoidal boundary with 8% independent flips. The learner is linear in both
tasks. Data, label noise and random acquisition use separate seeded generators.
Three seeds are paired comparisons, not 24 independent experiments. Linear and
curved tasks also share feature/noise draws within a seed, with different labels.

## At 40 labels

Values below are means ± sample standard deviations of held-out accuracy, in
percentage points; not confidence intervals or significance tests.

| Strategy | Linear | Curved |
|---|---:|---:|
| Random | 87.50 ± 0.78 | 83.20 ± 2.34 |
| Uncertainty | 90.23 ± 0.68 | 83.98 ± 3.73 |
| Diversity | 85.81 ± 3.99 | 82.55 ± 1.19 |
| Uncertainty + diversity | 88.80 ± 2.96 | 84.51 ± 0.60 |

Interpretation: uncertainty has the highest observed mean on the linear task;
combined acquisition leads on the curved task. Diversity alone trails random
in both. These are descriptive observations on a small fixed experiment, not a
general ordering. The misspecified curved model and label noise limit what useful
queries can accomplish. No hyperparameters were selected on the test set.

## Evidence and boundaries

`docs/evidence.json` stores six complete datasets and the 24 recorded acquisition
schedules, all selected IDs, 120 weights and metric snapshots, plus budget-wise
mean accuracy and sample standard deviation. Evaluation uses 256 test points per
run, fixed threshold logit >= 0; confusion counts determine accuracy, and stable
unregularized mean binary cross-entropy determines test log loss.

The browser does not perform acquisition. It loads a chosen recorded labeled-ID
list, validates it against the initial/pool universe and refits the classifier
from zero on only those points. Test rows enter scoring and optional visual
overlay only. The training selector rejects test IDs and missing initial IDs.
All synthetic labels are publicly inspectable for audit; this is not an access
security mechanism. Flipping all test labels leaves weights and acquisitions
unchanged in automated checks.

The probability field and point probes use newly computed browser weights.
The learning curves and three-seed aggregate summaries are saved Python results.
Every selected browser refit checks its weights/metrics against that snapshot.
Node tests cover all 120 cases at 1e-12 numerical tolerance; runtime UI parity
uses 1e-10. This is numerical agreement, not independent scientific replication.

Reproduce with `python export_demo.py --check`. Core and exporter SHA-256 bind
the source to canonical JSON; hashes identify content, not authenticated provenance.
No real-data study, latency benchmark, GPU computation, calibration evaluation,
statistical significance or competitive deep-learning result is claimed.
