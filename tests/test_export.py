import json
import unittest
from export_demo import TARGET, canonical, build
from active_lab import PoolPoint, score


class ExportTests(unittest.TestCase):
    def test_exact_source_bound_regeneration(self):
        self.assertEqual(TARGET.read_text(),canonical())

    def test_splits_and_recorded_schedule_conservation(self):
        data=build()
        self.assertEqual(len(data['datasets']),6)
        for d in data['datasets']:
            identities=[]
            for name,count in (('initial',8),('pool',96),('test',256)):
                self.assertEqual(len(d['splits'][name]),count)
                identities.extend(p['id'] for p in d['splits'][name])
            self.assertEqual(len(set(identities)),360)
            test=d['splits']['test']
            for run in data['report']['runs']:
                if (run['seed'],run['regime'])!=(d['seed'],d['regime']):continue
                initial=run['initial_ids']
                for c in run['curve']:
                    self.assertEqual(c['labeled_ids'],initial+run['acquired_ids'][:c['budget']-8])
                    m=score(c['weights'],[PoolPoint(p['id'],p['x']) for p in test],{p['id']:p['y'] for p in test})
                    self.assertEqual(m,c['test'])


if __name__=='__main__':unittest.main()
