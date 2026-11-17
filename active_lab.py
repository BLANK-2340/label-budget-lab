"""Label Budget Lab: original dependency-free CPU active-learning reference.

Repository https://github.com/BLANK-2340/label-budget-lab
Third November project, created public November 17, 2026 under BLANK-2340.
Run with Python 3.10+, no packages/data downloads/accounts:
    python -W error active_lab.py --self-test
    python active_lab.py

Scientific question: does querying uncertain or spatially diverse examples
improve held-out accuracy at a fixed annotation budget compared with random?
This is a bounded educational experiment, not a paper reproduction or novelty.

Architecture: deterministic original 2D synthetic data; three-parameter logistic
classifier; stable mean binary cross-entropy + L2 on weights (not bias); full
batch gradient descent from zero at EVERY budget. Acquisition sees feature-only
PoolPoint objects, never hidden labels or test data. A budgeted oracle reveals
only acquired initial/pool labels. Evaluation is downstream reporting only.

Protocol fixed before execution: seeds 17/42/2026; linear and curved boundaries;
8 initial labels, 96 unlabeled pool points, 256 independent test points; budgets
8/16/24/32/40 including initial labels; batches 8. Four policies: uniform random
without replacement, uncertainty |logit| ascending, farthest-first diversity in
raw input space, and uncertainty shortlist 3*batch then farthest-first. Ties use
IDs. Fit 400 iterations, learning rate .5, L2 .02. Raw features already in [-1,1].
No hyperparameter selection, feature normalization or stopping based on test.

Measured November17 at40labels, mean held-out accuracy over3seeds:
linear random .875000, uncertainty .902344, diversity .858073, combined .888021;
curved random .832031, uncertainty .839844, diversity .825521, combined .845052.
These small paired synthetic results do not establish a general advantage.
The JSON command records all runs, acquisitions, weights, curves, confusion
counts, log loss, mean accuracies and sample standard deviations at each budget.
Validation commands include finite-difference gradient and known metric tests,
atomic oracle/budget tests, exhaustive protocol counts, repeated-seed consistency
and a test-label-flip test preserving all acquisitions and fitted weights.

Linear boundary: y > .6*x; curved: y > .55*sin(pi*x). Oracle flips labels with
probability .08 using a separate deterministic generator. Classifier remains
linear on BOTH tasks, intentionally misspecified for the curved boundary.
Within a run all policies share exactly the same data and initial labels.
Acquisition RNG is separate from data/label RNG. Repeated policies are paired,
not independent experimental samples. Three seeds do not prove significance.

Current milestone: CPU implementation/tests/JSON learning curves only. No public
interactive demo, README, separate license file or CI yet; NOT showcase-ready.
Next: build interactive CPU/browser demonstration with fixed-budget controls,
decision-boundary/learning-curve explanations, source-bound reproducible export,
README/architecture/results/limitations/license/CI and live public acceptance.
Prioritize third showcase by November20 before tinyLM/CUDAfusion. Preserve older
repositories; do not rebuild the completed anomaly/tracking foundations.

All code and synthetic data original; no external code/assets incorporated.
MIT License. Copyright (c) 2026 Armaan Singh
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to
do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import unittest

POLICIES = ('random', 'uncertainty', 'diversity', 'uncertainty_diversity')
BUDGETS = (8, 16, 24, 32, 40)


@dataclass(frozen=True)
class PoolPoint:
    identity: str
    x: tuple

    def __post_init__(self):
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError('nonempty point ID required')
        if len(self.x) != 2 or any(not math.isfinite(v) or abs(v) > 1 for v in self.x):
            raise ValueError('two finite features in [-1,1] required')


def sigmoid(z):
    if not math.isfinite(z):
        raise ValueError('finite logit required')
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def logit(weights, point):
    return weights[0]*point.x[0]+weights[1]*point.x[1]+weights[2]


def objective(weights, labeled, l2=.02):
    if (len(weights) != 3 or any(not math.isfinite(w) for w in weights)
            or not labeled or not math.isfinite(l2) or l2 < 0):
        raise ValueError('finite weights, nonempty labels and nonnegative L2 required')
    loss, grad = 0., [0., 0., 0.]
    if len({p.identity for p, _ in labeled}) != len(labeled):
        raise ValueError('duplicate labeled IDs')
    for point, target in labeled:
        if type(target) is not int or target not in (0, 1):
            raise ValueError('integer binary label required')
        z = logit(weights, point)
        loss += max(z, 0)-z*target+math.log1p(math.exp(-abs(z)))
        residual = sigmoid(z)-target
        for j, feature in enumerate((*point.x, 1)):
            grad[j] += residual*feature
    count = len(labeled)
    return (loss/count+l2*(weights[0]**2+weights[1]**2)/2,
            tuple(g/count+(l2*weights[j] if j < 2 else 0) for j, g in enumerate(grad)))


def fit(labeled, steps=400, rate=.5, l2=.02):
    if type(steps) is not int or not 1 <= steps <= 5000 or not math.isfinite(rate) or rate <= 0:
        raise ValueError('bounded positive iterations and learning rate required')
    weights = (0., 0., 0.)
    for _ in range(steps):
        _, grad = objective(weights, labeled, l2)
        weights = tuple(w-rate*g for w, g in zip(weights, grad))
    objective(weights, labeled, l2)  # Reject nonfinite output before returning.
    return weights


def dataset(seed, regime):
    if type(seed) is not int or regime not in ('linear', 'curved'):
        raise ValueError('integer seed and known regime required')
    feature_rng, oracle_rng = random.Random(seed), random.Random(seed+1000000)
    splits, labels = {}, {}
    for name, count in (('initial', 8), ('pool', 96), ('test', 256)):
        rows = []
        for i in range(count):
            x = (feature_rng.uniform(-1, 1), feature_rng.uniform(-1, 1))
            point = PoolPoint(f'{name}:{i:03d}', x)
            boundary = .6*x[0] if regime == 'linear' else .55*math.sin(math.pi*x[0])
            target = int(x[1] > boundary)
            labels[point.identity] = target ^ int(oracle_rng.random() < .08)
            rows.append(point)
        splits[name] = tuple(rows)
    return splits, labels


class Oracle:
    def __init__(self, labels, allowed, cap=40):
        if type(cap) is not int or cap <= 0:
            raise ValueError('positive integer label cap required')
        self._labels, self.allowed, self.cap = dict(labels), frozenset(allowed), cap
        self.queried = set()

    def reveal(self, identities):
        ids = tuple(identities)
        if len(set(ids)) != len(ids) or any(i not in self.allowed or i in self.queried for i in ids):
            raise ValueError('oracle access must be new unique allowed IDs')
        if len(self.queried)+len(ids) > self.cap or any(i not in self._labels for i in ids):
            raise ValueError('oracle label cap exceeded or unknown ID')
        values = tuple(self._labels[i] for i in ids)
        self.queried.update(ids)
        return values


def acquire(pool, labeled_points, weights, policy, batch, rng):
    """Feature-only acquisition. No labels or evaluation set accepted here."""
    if policy not in POLICIES or type(batch) is not int or not 1 <= batch <= len(pool):
        raise ValueError('known policy and feasible positive batch required')
    if not labeled_points or len({p.identity for p in pool}) != len(pool):
        raise ValueError('nonempty labeled features and unique pool required')
    if {p.identity for p in pool} & {p.identity for p in labeled_points}:
        raise ValueError('labeled/pool overlap')
    if len(weights) != 3 or not all(math.isfinite(w) for w in weights):
        raise ValueError('finite model required')
    candidates = sorted(pool, key=lambda p: p.identity)
    if policy == 'random':
        return tuple(rng.sample(candidates, batch))
    ranked = sorted(candidates, key=lambda p: (abs(logit(weights, p)), p.identity))
    if policy == 'uncertainty':
        return tuple(ranked[:batch])
    if policy == 'uncertainty_diversity':
        candidates = ranked[:min(len(ranked), 3*batch)]
    selected, anchors = [], list(labeled_points)
    for _ in range(batch):
        point = min(candidates, key=lambda p: (-min(sum((x-y)**2 for x, y in zip(p.x, a.x))
                                                   for a in anchors), p.identity))
        selected.append(point)
        anchors.append(point)
        candidates.remove(point)
    return tuple(selected)


def score(weights, rows, labels):
    if not rows:
        raise ValueError('nonempty evaluation required')
    counts = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for p in rows:
        target = labels[p.identity]
        predicted = int(logit(weights, p) >= 0)
        counts['tp' if predicted and target else 'fp' if predicted else 'fn' if target else 'tn'] += 1
    loss, _ = objective(weights, [(p, labels[p.identity]) for p in rows], l2=0)
    return {**counts, 'accuracy': (counts['tp']+counts['tn'])/len(rows), 'log_loss': loss}


def run(seed, regime, policy, test_labels_override=None):
    if policy not in POLICIES:
        raise ValueError('unknown policy')
    splits, labels = dataset(seed, regime)
    allowed = {p.identity for key in ('initial', 'pool') for p in splits[key]}
    oracle = Oracle(labels, allowed)
    labeled_points, pool = list(splits['initial']), list(splits['pool'])
    targets = list(oracle.reveal([p.identity for p in labeled_points]))
    rng = random.Random(seed+2000000)
    selected, curve = [], []
    test_labels = labels if test_labels_override is None else test_labels_override
    for budget in BUDGETS:
        if len(labeled_points) != budget:
            raise RuntimeError('budget protocol violated')
        weights = fit(list(zip(labeled_points, targets)))
        curve.append({'budget': budget, 'weights': weights,
                      'labeled_ids': [p.identity for p in labeled_points],
                      'test': score(weights, splits['test'], test_labels)})
        if budget == BUDGETS[-1]:
            break
        chosen = acquire(pool, labeled_points, weights, policy, 8, rng)
        targets.extend(oracle.reveal([p.identity for p in chosen]))
        labeled_points.extend(chosen)
        chosen_ids = {p.identity for p in chosen}
        selected.extend(p.identity for p in chosen)
        pool = [p for p in pool if p.identity not in chosen_ids]
    return {'seed': seed, 'regime': regime, 'policy': policy,
            'initial_ids': [p.identity for p in splits['initial']],
            'test_ids': [p.identity for p in splits['test']],
            'acquired_ids': selected, 'oracle_labels_spent': len(oracle.queried), 'curve': curve}


def experiment():
    runs = [run(seed, regime, policy) for seed in (17, 42, 2026)
            for regime in ('linear', 'curved') for policy in POLICIES]
    aggregate = []
    for regime in ('linear', 'curved'):
        for policy in POLICIES:
            group = [r for r in runs if r['regime'] == regime and r['policy'] == policy]
            aggregate.append({'regime': regime, 'policy': policy, 'curve': [
                {'budget': budget, 'accuracy_mean': statistics.mean(r['curve'][i]['test']['accuracy'] for r in group),
                 'accuracy_sample_stddev': statistics.stdev(r['curve'][i]['test']['accuracy'] for r in group)}
                for i, budget in enumerate(BUDGETS)]})
    return {'schema': 1, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'original synthetic 2D logistic active learning; no universal advantage claim',
            'runs': runs, 'aggregate': aggregate}


class Tests(unittest.TestCase):
    def test_stable_loss_extreme_logits(self):
        p = PoolPoint('a', (1., 0.))
        self.assertEqual(sigmoid(1000), 1.)
        self.assertEqual(sigmoid(-1000), 0.)
        loss, grad = objective((1000, 0, 0), [(p, 0)], l2=0)
        self.assertEqual(loss, 1000.)
        self.assertEqual(grad, (1., 0., 1.))

    def test_gradient_central_difference(self):
        labeled = [(PoolPoint('a', (.2, -.8)), 1), (PoolPoint('b', (-.7, .3)), 0)]
        weights = (.4, -.9, .1)
        for l2 in (0, .02):
            _, grad = objective(weights, labeled, l2)
            for j in range(3):
                plus, minus = list(weights), list(weights)
                plus[j] += 1e-5
                minus[j] -= 1e-5
                numeric = (objective(plus,labeled,l2)[0]-objective(minus,labeled,l2)[0])/2e-5
                self.assertAlmostEqual(grad[j], numeric, places=8)

    def test_training_and_known_metrics(self):
        labeled = [(PoolPoint('a', (-1., 0.)), 0), (PoolPoint('b', (1., 0.)), 1)]
        w = fit(labeled)
        self.assertLess(objective(w, labeled)[0], objective((0,0,0),labeled)[0])
        m = score(w, [p for p,_ in labeled], {'a':0,'b':1})
        self.assertEqual((m['tp'],m['tn'],m['fp'],m['fn'],m['accuracy']), (1,1,0,0,1.))
        self.assertEqual(w, fit(labeled))

    def test_known_acquisition_geometry_and_ties(self):
        anchor = [PoolPoint('initial:a', (0.,0.))]
        pool = [PoolPoint('pool:a', (.1,0.)), PoolPoint('pool:b', (.9,0.)), PoolPoint('pool:c', (-.9,0.))]
        self.assertEqual(acquire(pool,anchor,(1,0,0),'uncertainty',1,random.Random(1))[0].identity,'pool:a')
        self.assertEqual([p.identity for p in acquire(pool,anchor,(1,0,0),'diversity',2,random.Random(1))],['pool:b','pool:c'])
        self.assertEqual(acquire(pool,anchor,(0,0,0),'uncertainty',1,random.Random(1))[0].identity,'pool:a')

    def test_oracle_budget_and_atomic_failure(self):
        oracle = Oracle({'a':0,'b':1,'test':1}, {'a','b'}, cap=1)
        for ids in [('test',), ('a','a'), ('a','b')]:
            with self.assertRaises(ValueError):
                oracle.reveal(ids)
            self.assertFalse(oracle.queried)
        self.assertEqual(oracle.reveal(['a']), (0,))
        with self.assertRaises(ValueError):
            oracle.reveal(['a'])

    def test_seed_splits_and_budget_no_repeated_queries(self):
        splits, _ = dataset(42,'curved')
        sets = [{p.identity for p in splits[k]} for k in ('initial','pool','test')]
        self.assertEqual([len(s) for s in sets], [8,96,256])
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(dataset(42,'curved'), dataset(42,'curved'))
        initial = []
        for policy in POLICIES:
            r = run(42,'curved',policy)
            self.assertEqual(r['oracle_labels_spent'],40)
            self.assertEqual(len(set(r['acquired_ids'])),32)
            self.assertFalse(set(r['acquired_ids'])&sets[2])
            self.assertEqual([c['budget'] for c in r['curve']],list(BUDGETS))
            for c in r['curve']:
                self.assertEqual(len(set(c['labeled_ids'])),c['budget'])
            initial.append(r['curve'][0]['weights'])
        self.assertTrue(all(w==initial[0] for w in initial))

    def test_test_label_changes_do_not_change_training_or_acquisition(self):
        splits, labels = dataset(17,'linear')
        flipped = {p.identity:1-labels[p.identity] for p in splits['test']}
        a = run(17,'linear','uncertainty_diversity')
        b = run(17,'linear','uncertainty_diversity',flipped)
        self.assertEqual(a['acquired_ids'], b['acquired_ids'])
        self.assertEqual([c['weights'] for c in a['curve']],[c['weights'] for c in b['curve']])
        for x,y in zip(a['curve'],b['curve']):
            self.assertEqual(x['test']['accuracy']+y['test']['accuracy'],1.)

    def test_feature_only_combined_shortlist_and_no_mutation(self):
        anchor = [PoolPoint('initial:a',(0.,0.))]
        pool = [PoolPoint(f'pool:{i}',(i/10,0.)) for i in range(1,10)]
        before = tuple(pool)
        selected = acquire(pool,anchor,(1,0,0),'uncertainty_diversity',2,random.Random(1))
        self.assertEqual([p.identity for p in selected],['pool:6','pool:3'])
        self.assertEqual(tuple(pool), before)
        self.assertEqual(set(PoolPoint.__dataclass_fields__), {'identity','x'})
        a = acquire(pool,anchor,(1,0,0),'random',8,random.Random(9))
        b = acquire(pool,anchor,(1,0,0),'random',8,random.Random(9))
        self.assertEqual(a,b)
        self.assertEqual(len(set(p.identity for p in a)),8)

    def test_full_protocol_and_independent_aggregate(self):
        report = experiment()
        self.assertEqual(len(report['runs']),24)
        self.assertEqual(len(report['aggregate']),8)
        for r in report['runs']:
            for c in r['curve']:
                m = c['test']
                self.assertEqual(m['tp']+m['tn']+m['fp']+m['fn'],256)
                self.assertEqual(m['accuracy'],(m['tp']+m['tn'])/256)
                self.assertFalse(set(c['labeled_ids'])&set(r['test_ids']))
        for group in report['aggregate']:
            runs = [r for r in report['runs'] if r['regime']==group['regime'] and r['policy']==group['policy']]
            for i,c in enumerate(group['curve']):
                values = [r['curve'][i]['test']['accuracy'] for r in runs]
                mean = sum(values)/3
                deviation = math.sqrt(sum((v-mean)**2 for v in values)/2)
                self.assertAlmostEqual(c['accuracy_mean'],mean)
                self.assertAlmostEqual(c['accuracy_sample_stddev'],deviation)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            PoolPoint('x',(math.nan,0))
        with self.assertRaises(ValueError):
            fit([])
        with self.assertRaises(ValueError):
            objective((0,0,0),[(PoolPoint('a',(0,0)),2)])
        with self.assertRaises(ValueError):
            acquire([],[],(0,0,0),'unknown',1,random.Random(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        unittest.main(argv=[__file__],exit=True)
    else:
        print(json.dumps(experiment(),sort_keys=True,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
