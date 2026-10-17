"""Dependency-free smoke tests for E4 algebra, subsetting and aggregators."""
import importlib.util,json,math,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'experiments/04_backend_fidelity/scripts'

def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

audit=load('audit_e1_logits')
runner=load('run_e4_paired')
summ=load('summarize_e4')

class E4Tests(unittest.TestCase):
    def test_prob_exact_large_negative(self):
        for a,b in ((-4,-1),(-1001,-1010),(-.01,-12),(-7,-7)):
            q=audit.softmax_pair(a,b)
            self.assertTrue(0<=q<=1)
            self.assertAlmostEqual(q,1/(1+math.exp(b-a)),places=12)
    def test_subset_54(self):
        rows=[{'id':f'{typ}-{i}','question':{'type':typ}} for typ,count in [('choice',139),('noul',74),('score',18)] for i in range(count)]
        selected,tag=runner.subset_tasks(rows,False,18)
        self.assertEqual(tag,'balanced18_seed42')
        self.assertEqual(len(selected),54)
        self.assertEqual(runner.subset_tasks(rows,True,5)[1],'full')
    def test_audit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ollama2jev_jevbench_test_full.jsonl'
            good={'id':'one','diagnostics':[{'yes_logprob':-1.,'no_logprob':-2.,'p_yes':audit.softmax_pair(-1,-2)}]}
            missing={'id':'bad','diagnostics':[{'yes_logprob':-1.,'no_logprob':None,'p_yes':None}],'error':'no token'}
            p.write_text('\n'.join(json.dumps(x) for x in ([good]*230+[missing]))+'\n')
            q=audit.audit_file(p)
            self.assertEqual(q['tasks'],231)
            self.assertEqual(q['missing'],1)
            self.assertEqual(q['maximum_absolute_probability_error'],0)
    def test_summarize(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'e4_a_balanced1_r1.jsonl'
            a={'status':'ok','latency_s':.1,'candidate_p_yes':[.8,.3],
                'predicted':{'expected':'A','prediction':'A','correct':True,'probabilities':{'A':.9,'B':.1}}}
            b={'status':'ok','latency_s':.2,'candidate_p_yes':[.7,.4],
                'predicted':{'expected':'A','prediction':'A','correct':True,'probabilities':{'A':.8,'B':.2}}}
            p.write_text(json.dumps({'id':'i','repetition':0,'config_sha256':'test','ollama':a,'transformers':b})+'\n')
            z=summ.analyze(p)
            self.assertEqual(z['paired_binary_candidates'],2)
            self.assertAlmostEqual(z['binary_probability_mae'],.1)
            self.assertEqual(z['typed_prediction_agreement'],1.)

if __name__=='__main__':unittest.main()
