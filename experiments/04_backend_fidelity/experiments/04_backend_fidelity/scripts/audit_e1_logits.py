#!/usr/bin/env python3
"""E4A deterministic numerical audit from E1 Ollama top-20 logprob JSONLs.

This confirms exact reconstruction of the CONDITIONAL two-token probability
from the two observed log-probabilities. It does NOT establish cross-runtime
fidelity to a distinct full-vocabulary model.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E1 = ROOT / 'experiments/01_general_llm_benchmark/results/raw'
OUT = ROOT / 'experiments/04_backend_fidelity/results/offline'
EXPECTED = 11

def softmax_pair(log_yes, log_no):
    center = max(log_yes, log_no)
    ey, en = math.exp(log_yes-center), math.exp(log_no-center)
    return ey/(ey+en)

def audit_file(path):
    n_tasks = ok = failed = n_seen = covered = missing = 0
    diffs = []
    bad_ids = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n_tasks += 1
        if row.get('error'):
            failed += 1
        else:
            ok += 1
        for d in row.get('diagnostics', []):
            n_seen += 1
            yes, no, p = (d.get(x) for x in ('yes_logprob','no_logprob','p_yes'))
            if yes is None or no is None or p is None:
                missing += 1
                bad_ids.append(row['id'])
                continue
            actual = softmax_pair(float(yes),float(no))
            if not (math.isfinite(actual) and math.isfinite(float(p))):
                raise ValueError(f'Nonfinite diagnostic in {path}: {row["id"]}')
            diffs.append(abs(actual-float(p)))
            covered += 1
    if n_tasks != 231:
        raise ValueError(f'{path}: expected 231 rows, got {n_tasks}')
    return dict(model=path.name.removeprefix('ollama2jev_jevbench_').removesuffix('_full.jsonl'),
                tasks=n_tasks, complete=ok, failed=failed,
                candidate_attempts=n_seen, covered=covered, missing=missing,
                coverage=covered/n_seen if n_seen else None,
                mean_absolute_probability_error=sum(diffs)/len(diffs) if diffs else None,
                maximum_absolute_probability_error=max(diffs) if diffs else None,
                uncovered_task_ids=';'.join(bad_ids))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--e1-raw',type=Path,default=E1)
    ap.add_argument('--out-dir',type=Path,default=OUT)
    args=ap.parse_args()
    files=sorted(args.e1_raw.glob('ollama2jev_jevbench_*_full.jsonl'))
    if len(files)!=EXPECTED: raise ValueError(f'Expected {EXPECTED} per-model E1 JSONLs, found {len(files)} at {args.e1_raw}')
    rows=[audit_file(p) for p in files]
    totals={key:sum(row[key] for row in rows) for key in ('tasks','complete','failed','candidate_attempts','covered','missing')}
    totals['coverage']=totals['covered']/totals['candidate_attempts']
    totals['maximum_absolute_probability_error']=max(r['maximum_absolute_probability_error'] for r in rows)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    with (args.out_dir/'e4a_e1_logprob_audit.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (args.out_dir/'e4a_e1_logprob_audit.json').write_text(json.dumps({'method':'algebraic replay of E1 first-token yes/no logprobs','limitation':'NOT a comparison with independent full-vocabulary exact logits','rows':rows,'total':totals},indent=2)+'\n',encoding='utf-8')
    print('Models:',len(rows),'decisions:',totals['tasks'],'binary attempts:',totals['candidate_attempts'])
    print('Covered:',totals['covered'],'missing:',totals['missing'],'coverage:',f"{totals['coverage']:.6f}")
    print('Maximum absolute p_yes reconstruction error:',f"{totals['maximum_absolute_probability_error']:.3g}")
    print('Saved:',args.out_dir)
if __name__=='__main__':main()
