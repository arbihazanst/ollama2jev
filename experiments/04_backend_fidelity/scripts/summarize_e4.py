#!/usr/bin/env python3
"""Summarize E4B paired attempts (including full-denominator failures)."""
from __future__ import annotations
import argparse,csv,json,math,statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_IN=ROOT/'experiments/04_backend_fidelity/results/paired/raw'
DEFAULT_OUT=ROOT/'experiments/04_backend_fidelity/results/paired/summary'

def pct(xs,p):
    if not xs:return None
    xs=sorted(xs);pos=(len(xs)-1)*p;lo=int(pos);hi=min(lo+1,len(xs)-1)
    return xs[lo]+(xs[hi]-xs[lo])*(pos-lo)

def mean(xs):return statistics.mean(xs) if xs else None

def quality(rows,backend):
    good=[r[backend] for r in rows if r.get(backend) and r[backend]['status']=='ok']
    if not good:return {'n':0,'correct':0,'accuracy_full':0,'brier_valid':None,'nll_valid':None,'ece10_valid':None}
    losses=[];nll=[];preds=[]
    for a in good:
        z=a['predicted'];gold=z['expected'];probs=z['probabilities'];conf=max(probs.values())
        losses.append(sum((p-(1. if lab==gold else 0.))**2 for lab,p in probs.items()))
        nll.append(-math.log(max(probs[gold],1e-15)))
        preds.append((conf,int(z['correct'])))
    ece=0
    for i in range(10):
        box=[(p,a) for p,a in preds if min(9,int(p*10))==i]
        if box:ece+=len(box)/len(preds)*abs(mean([x[0] for x in box])-mean([x[1] for x in box]))
    return {'n':len(good),'correct':sum(z['predicted']['correct'] for z in good),'accuracy_full':sum(z['predicted']['correct'] for z in good)/len(rows),
            'brier_valid':mean(losses),'nll_valid':mean(nll),'ece10_valid':ece}

def analyze(path):
    record={}
    for num,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        try:r=json.loads(line)
        except json.JSONDecodeError as e:raise ValueError(f'{path}:{num}: invalid JSON') from e
        key=(r['id'],r['repetition'])
        if key in record and r['config_sha256']!=record[key]['config_sha256']:
            raise ValueError(f'Conflicting E4 run config in {path}:{num}')
        record[key]=r
    rows=list(record.values());rows.sort(key=lambda r:(r['id'],r['repetition']))
    if not rows:raise ValueError(f'No entries in {path}')
    configs={x['config_sha256'] for x in rows}
    if len(configs)!=1:raise ValueError(f'Mixed configs in {path}')
    n_selected=len({r['id'] for r in rows});reps=1+max(r['repetition'] for r in rows)
    expected=n_selected*reps
    first=[r for r in rows if r['repetition']==0]
    candidate_pairs=[];typed_pairs=[];token_parity=[];latencies={};backend_status={}
    for backend in ('ollama','transformers'):
        values=[r[backend] for r in rows if r.get(backend)]
        backend_status[backend]=dict((name,sum(x['status']==name for x in values)) for name in ('ok','coverage_failure','error'))
        latencies[backend]=[x['latency_s'] for x in values if x['status']=='ok' and isinstance(x.get('latency_s'),(int,float))]
    for r in rows:
        a=r.get('ollama');b=r.get('transformers')
        if a and b and a['status']=='ok' and b['status']=='ok':
            typed_pairs.append((a['predicted']['prediction']==b['predicted']['prediction'],a['latency_s'],b['latency_s']))
            candidate_pairs.extend(zip(a['candidate_p_yes'],b['candidate_p_yes']))
            if r.get('comparison') and 'prompt_token_count_match' in r['comparison']:
                token_parity.append(bool(r['comparison']['prompt_token_count_match']))
    absd=[abs(a-b) for a,b in candidate_pairs]
    ollama=[a for a,b in candidate_pairs];hf=[b for a,b in candidate_pairs]
    pearson=None
    if len(candidate_pairs)>1:
        av,bv=mean(ollama),mean(hf)
        cov=sum((a-av)*(b-bv) for a,b in candidate_pairs)
        va=sum((a-av)**2 for a in ollama);vb=sum((b-bv)**2 for b in hf)
        if va>0 and vb>0:pearson=cov/math.sqrt(va*vb)
    ratios=[a/b for _,a,b in typed_pairs if b>0]
    result={
        'source_jsonl':path.name,'status':'complete' if len(rows)==expected and len(first)==n_selected else 'partial',
        'distinct_tasks':n_selected,'repetitions':reps,'recorded_model_task_repetitions':len(rows),
        'expected_model_task_repetitions':expected,'first_repetition_rows':len(first),
        'config_sha256':next(iter(configs)),'only_ollama':all(not r.get('transformers') for r in rows),
        'ollama_status':backend_status['ollama'],'transformers_status':backend_status['transformers'],
        'ollama_first_run_quality':quality(first,'ollama'),
        'hf_first_run_quality':quality(first,'transformers'),
        'paired_type_decisions':len(typed_pairs),'paired_binary_candidates':len(candidate_pairs),
        'binary_probability_mae':mean(absd),'binary_probability_rmse':math.sqrt(mean([x*x for x in absd])) if absd else None,
        'binary_probability_max_abs_error':max(absd) if absd else None,
        'binary_probability_pearson':pearson,
        'typed_prediction_agreement':mean([int(x[0]) for x in typed_pairs]),
        'prompt_token_count_match_rate':mean([int(x) for x in token_parity]),
        'ollama_request_mean_s':mean(latencies['ollama']),'ollama_request_p50_s':pct(latencies['ollama'],.5),
        'ollama_request_p95_s':pct(latencies['ollama'],.95),
        'hf_request_mean_s':mean(latencies['transformers']),'hf_request_p50_s':pct(latencies['transformers'],.5),
        'hf_request_p95_s':pct(latencies['transformers'],.95),
        'paired_latencies_ratio_ollama_over_hf_mean':mean(ratios),
        'limitations':'These are cross-stack measurements unless weights, tokenizer IDs and rendered chat-template tokens are independently verified identical. HF full-logit reference can use different quantization from Ollama GGUF. Warmup/model loading excluded; CPU/GPU contention may affect timing.'
    }
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=DEFAULT_IN)
    p.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    args=p.parse_args()
    paths=sorted(args.raw_dir.glob('e4_*.jsonl'))
    if not paths:raise FileNotFoundError(f'No E4 JSONL files at {args.raw_dir}')
    results=[analyze(path) for path in paths]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/'e4_cross_stack_summary.csv').open('w',newline='',encoding='utf-8') as f:
        # Nested values compactly JSON-encode for CSV.
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader()
        for r in results:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in r.items()})
    (args.out_dir/'e4_cross_stack_summary.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    for r in results:
        print(f"{r['source_jsonl']}: {r['status']}, {r['paired_binary_candidates']} paired binary decisions, MAE={r['binary_probability_mae']}, ollama/HF P50={r['ollama_request_p50_s']}/{r['hf_request_p50_s']}")
    print('Saved',args.out_dir)
if __name__=='__main__':main()
