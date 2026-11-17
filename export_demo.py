"""Export fixed acquisition schedules; browser refits/inference are live CPU."""
import argparse
import hashlib
import json
from pathlib import Path
from active_lab import dataset, experiment

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / 'docs/evidence.json'


def build():
    report = experiment()
    datasets = []
    for seed in (17,42,2026):
        for regime in ('linear','curved'):
            splits, labels = dataset(seed, regime)
            datasets.append({'seed':seed, 'regime':regime, 'splits':{
                name:[{'id':p.identity,'x':p.x,'y':labels[p.identity]} for p in rows]
                for name,rows in splits.items()}})
    return {'schema':1,'mode':'live CPU refit; recorded acquisition schedules',
            'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                             for p in ('active_lab.py','export_demo.py')},
            'datasets':datasets,'report':report}


def canonical():
    return json.dumps(build(),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    content=canonical()
    if args.check:
        if TARGET.read_text(encoding='utf-8')!=content:
            raise SystemExit('source/protocol evidence differs from exact recomputation')
        print('source-bound datasets and 24 acquisition schedules reproduce exactly')
    else:
        TARGET.parent.mkdir(exist_ok=True)
        TARGET.write_text(content,encoding='utf-8')


if __name__=='__main__':
    main()
