# Label Budget Lab

Original, dependency-free CPU active-learning workbench. Ask whether uncertainty
or spatial diversity helps a classifier at a fixed annotation budget. Examine
both gains and failures on a linear task and a deliberately misspecified curved
task. This is a bounded educational experiment, not a paper reproduction.

[Open the public demo](https://blank-2340.github.io/label-budget-lab/).
Deployment and core interactions must be verified separately from publication;
the latest delivery report records actual live acceptance.

## Run

Python 3.10+, no packages, external data, paid APIs or accounts required locally.
Node 22+ is used for browser-reference checks, with no npm packages.

```bash
python -W error active_lab.py --self-test
python active_lab.py
python -m http.server 8000
```

Open `http://localhost:8000/docs/`. Choose problem, seed, acquisition strategy and
total label budget. The classifier **refits from zero in your browser** on the
selected acquired labels. Show held-out test points without training on them;
click the probability field to run live prediction at a new point.

Acquisition schedules and datasets are frozen Python exports. The learning curves
are saved experiment results, not live acquisition or repeated new trials. The
public synthetic bundle contains all labels for audit, but only acquired labels
are passed to training. This is a protocol boundary, not a data-secrecy claim.

## Architecture and fixed protocol

- Three-parameter logistic regression: stable mean binary cross-entropy plus
  L2=.02 on two weights, no bias regularization. Reinitialize to zero and run 400
  full-batch gradient steps at learning rate .5 at every budget.
- Seeds 17/42/2026; 8 initial labels, 96 pool features, 256 independent test
  observations. Budgets 8/16/24/32/40 include the initial 8; query batches of 8.
- Original features uniform in [-1,1]². True boundaries: y>.6*x or
  y>.55*sin(pi*x). Oracle independently flips labels with probability .08.
- Random without replacement; margin uncertainty; farthest-first raw-feature
  diversity; or uncertainty shortlist of 3×batch followed by diversity. Ties use
  IDs. No hidden labels or test data enter the acquisition function.
- A guarded oracle allows new initial/pool IDs only and caps labels at 40.
  Test labels enter reporting only; no tuning, normalization or stopping on test.
- Browser refitting and inference are checked against all 120 Python models
  (24 runs × 5 budgets), including weights, confusion counts, accuracy and loss.

The curved boundary is intentionally not representable by this linear model.
See [results and limitations](RESULTS.md) for the full measurement scope.

## Recorded results

Held-out accuracy at 40 labels, mean over three paired seeds:

| Strategy | Linear | Curved |
|---|---:|---:|
| Random | 87.50% | 83.20% |
| Uncertainty | 90.23% | 83.98% |
| Diversity | 85.81% | 82.55% |
| Uncertainty + diversity | 88.80% | 84.51% |

Different strategies lead on different tasks; diversity alone trails random on
both. Three seeds do not establish significance or a universal label-efficiency
advantage. Individual seed/budget results can be worse than random.

## Tests and reproducibility

```bash
python -W error active_lab.py --self-test
python -W error -m unittest discover -s tests -v
python -W error export_demo.py --check
node --check docs/app.js
node --test tests/engine.test.cjs
```

`python export_demo.py` deliberately regenerates `docs/evidence.json`. `--check`
requires byte-identical recomputation, including core/exporter SHA-256 identities.
CI runs these commands. Tests include central finite differences, stable extreme
logits, known geometry/metrics, oracle atomicity, budget and split conservation,
test-label-flip invariance, acquisition uniqueness, aggregate sample deviation,
cross-language parity, UI controls and plot-coordinate orientation.

## Limitations, source and license

Tiny original 2D synthetic tasks only. No real-data, GPU speed, modern deep active
learning, calibrated confidence, novelty or state-of-the-art claim. Model output
probabilities are not a calibration guarantee. Recorded policy selections cannot
be changed by interactions. Mobile browser validation remains pending.

All code, generated data and site assets are original. No external code/data/assets
were incorporated; no paper reproduction claimed. MIT license. Read
[persistent continuity](PROGRESS.md); historical module status is superseded there.
