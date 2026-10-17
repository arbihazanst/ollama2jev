#!/usr/bin/env python3
"""Offline Z1/Z2/Z3/Z4 ablations for the published Ollama2Jev JevBench JSONL.

Requires only Python 3.10+ stdlib and the ORIGINAL JevBench dataset files.
No Ollama calls are made and no run data is overwritten.

Typical invocation from Ollama2Jev repository root:
 python reproduce_z2_z4_ablation.py \
   --results-dir . --dataset-dir ../jevbench/datasets/public \
   --out-dir ablation_offline
"""
import argparse
import csv
import json
import math
from pathlib import Path

MODELS = [
    'qwen3.5:0.8b', 'qwen3.5:2b', 'qwen3.5:4b', 'qwen3.5:9b',
    'qwen3.8:latest', 'llama3.2:1b', 'llama3.2:3b',
    'granite4.2:3b', 'granite4.2:8b', 'gemma4:e2b', 'gemma4:e4b',
]
KS = (2, 5, 10, 20)


def normalize(values):
    total = math.fsum(values)
    if total == 0:
        return [1 / len(values)] * len(values)
    return [x / total for x in values]


def stable_normalize_logprobs(logps):
    maxi = max(logps)
    if not math.isfinite(maxi):
        if maxi == -math.inf:
            return [1 / len(logps)] * len(logps)
        raise ValueError('invalid logprob')
    return normalize([math.exp(lp - maxi) for lp in logps])


def get_distribution(row, spec, variant):
    ds = row['diagnostics']
    if variant == 'Z1':
        raw = [d['p_yes'] for d in ds]
    elif variant == 'Z2':
        raw = [math.exp(d['yes_logprob']) for d in ds]
    elif variant == 'Z3':
        raw = [float(d['p_yes'] >= 0.5) for d in ds]
    else:
        raise ValueError(variant)
    if any(x is None or not math.isfinite(x) for x in raw):
        raise ValueError('nonfinite candidate score')

    kind = row['question_type']
    if kind == 'noul':
        if len(raw) == 1:
            true_prob = raw[0]
        elif len(raw) == 2:
            norm = (stable_normalize_logprobs([d['yes_logprob'] for d in ds])
                    if variant == 'Z2' else normalize(raw))
            true_prob = norm[0]
        else:
            raise ValueError(f'Unexpected Noul candidate count: {len(raw)}')
        probs, labels = [true_prob, 1.0 - true_prob], ['yes', 'no']
    else:
        if kind == 'choice':
            labels = list(spec['question']['criteria'])
        elif kind == 'score':
            labels = list(range(len(raw)))
        else:
            raise ValueError(f'unknown decision primitive: {kind}')
        if len(labels) != len(raw):
            raise ValueError('choice/score candidate count mismatch')
        probs = (stable_normalize_logprobs([d['yes_logprob'] for d in ds])
                 if variant == 'Z2' else normalize(raw))

    # LLM2Jev picks the first maximum, and Noul thresholds at >= 0.5.
    predicted = labels[max(range(len(probs)), key=probs.__getitem__)]
    gold = row['expected']
    if gold not in labels:
        raise ValueError(f'gold not in labels: {gold!r} not in {labels!r}')
    gold_i = labels.index(gold)
    confidence = max(probs)
    brier = math.fsum((p - (j == gold_i)) ** 2 for j, p in enumerate(probs))
    nll = -math.log(max(probs[gold_i], 1e-15))
    return {
        'correct': predicted == gold, 'brier': brier, 'nll': nll,
        'confidence': confidence, 'primitive': kind,
    }


def ece10(records):
    n = len(records)
    if not n:
        return None
    total = 0.0
    for idx in range(10):
        b = [r for r in records if min(9, int(r['confidence'] * 10)) == idx]
        if b:
            observed = sum(r['correct'] for r in b) / len(b)
            predicted = math.fsum(r['confidence'] for r in b) / len(b)
            total += len(b) / n * abs(observed - predicted)
    return total


def exact_mcnemar(a_only, b_only):
    n = a_only + b_only
    if not n:
        return 1.0
    # Stable binomial summation via integer arithmetic.
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n)


def load_jsonl(path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', type=Path, default=Path('.'))
    ap.add_argument('--dataset-dir', type=Path, default=Path('../jevbench/datasets/public'))
    ap.add_argument('--out-dir', type=Path, default=Path('ablation_offline'))
    args = ap.parse_args()
    task_list = []
    for filename in ['original.jsonl', 'easy.jsonl', 'hard.jsonl']:
        task_list += load_jsonl(args.dataset_dir / filename)
    assert len(task_list) == 231, f'Expected 231 JevBench tasks, found {len(task_list)}'
    task_by_id = {task['id']: task for task in task_list}
    assert len(task_by_id) == 231

    main_rows = []
    coverage_rows = []
    paired_rows = []
    overall = {v: {'n': 0, 'correct': 0, 'brier_sum': 0.0} for v in ('Z1', 'Z2', 'Z3')}
    k_totals = {k: {'candidate': 0, 'attempted': 0, 'typed': 0, 'correct': 0} for k in KS}
    for model in MODELS:
        fname = f'ollama2jev_jevbench_{model.replace(":", "-")}_full.jsonl'
        rows = load_jsonl(args.results_dir / fname)
        assert len(rows) == 231, (fname, len(rows))
        assert set(r['id'] for r in rows) == set(task_by_id), f'ID mismatch {fname}'
        results = {}
        task_correct = {}
        for variant in ['Z1', 'Z2', 'Z3']:
            recs = []
            for row in rows:
                if 'prediction' not in row:
                    continue
                result = get_distribution(row, task_by_id[row['id']], variant)
                if variant == 'Z1' and result['correct'] != row['correct']:
                    raise AssertionError(f'Z1 reproduction failed for {model}/{row["id"]}')
                recs.append(result)
                task_correct.setdefault(row['id'], {})[variant] = result['correct']
            results[variant] = recs
            n = len(recs)
            correct = sum(r['correct'] for r in recs)
            brier = math.fsum(r['brier'] for r in recs) / n
            for col, val in [
                ('model', model), ('variant', variant), ('successful', n),
                ('correct', correct), ('accuracy_full', correct / 231),
                ('brier_success', brier), ('ece10_success', ece10(recs)),
                ('nll_success', math.fsum(r['nll'] for r in recs) / n),
            ]:
                pass
            main_rows.append({
                'model': model, 'variant': variant, 'successful': n,
                'correct': correct, 'accuracy_full': correct / 231,
                'brier_success': brier, 'ece10_success': ece10(recs),
                'nll_success': math.fsum(r['nll'] for r in recs) / n,
            })
            overall[variant]['n'] += n
            overall[variant]['correct'] += correct
            overall[variant]['brier_sum'] += brier * n
        for variant in ['Z2', 'Z3']:
            a_only = sum(v['Z1'] and not v[variant] for v in task_correct.values())
            b_only = sum(not v['Z1'] and v[variant] for v in task_correct.values())
            paired_rows.append({'model': model, 'contrast': 'Z1 versus ' + variant,
                'z1_only': a_only, 'ablation_only': b_only,
                'mcnemar_exact_two_sided_p': exact_mcnemar(a_only, b_only)})

        for k in KS:
            attempts = [d for r in rows for d in r.get('diagnostics', [])]
            candidate_eligible = sum(d.get('yes_rank') is not None and d.get('no_rank') is not None
                and d['yes_rank'] <= k and d['no_rank'] <= k for d in attempts)
            eligible = [r for r in rows if 'prediction' in r and all(
                d.get('yes_rank') is not None and d.get('no_rank') is not None
                and d['yes_rank'] <= k and d['no_rank'] <= k for d in r['diagnostics'])]
            correct = sum(r['correct'] for r in eligible)
            coverage_rows.append({
                'model': model, 'k': k, 'observed_candidate_coverage': candidate_eligible,
                'observed_candidates_attempted': len(attempts),
                'typed_decisions_covered': len(eligible), 'typed_decisions_total': 231,
                'correct_full_denominator': correct, 'accuracy_full': correct / 231,
                'accuracy_covered': correct / len(eligible) if eligible else None,
            })
            kt = k_totals[k]
            kt['candidate'] += candidate_eligible
            kt['attempted'] += len(attempts)
            kt['typed'] += len(eligible)
            kt['correct'] += correct

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [('z1_z3_model_metrics.csv', main_rows),
                       ('z4_model_coverage.csv', coverage_rows),
                       ('paired_z1_vs_ablations.csv', paired_rows)]:
        with (args.out_dir / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    payload = {'source_models': MODELS, 'successful_probability_metrics': overall,
               'topk_replay': k_totals, 'protocol': 'offline_replay_only',
               'n_models': 11, 'items_per_model': 231}
    (args.out_dir / 'ablation_summary.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print('Saved result CSV/JSON to', args.out_dir)


if __name__ == '__main__':
    main()
