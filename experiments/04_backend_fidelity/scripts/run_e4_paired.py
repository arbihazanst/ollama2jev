#!/usr/bin/env python3
"""E4B: paired Ollama top-20 versus Transformers full-logit reference.

Reuses the SAME binary prompt messages and LLM2Jev assembler on both paths.
IMPORTANT: HF and Ollama models may differ in quantization, template, runtime,
position/special tokens. Do not call this exact same-weight logit fidelity.
"""
from __future__ import annotations
import argparse, hashlib, json, os, platform, random, statistics, subprocess, sys, time
from urllib.request import urlopen
from collections import Counter
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_DATASET=ROOT.parent/'jevbench/datasets/public'
DEFAULT_OUT=ROOT/'experiments/04_backend_fidelity/results/paired'
PROTOCOL='e4-cross-stack-v1'
EXPECTED={'choice':139,'noul':74,'score':18}

def sha_json(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()

def safe(s):
    return ''.join(c if c.isalnum() or c in '.-_' else '-' for c in s)

def tasks_from_dataset(folder):
    rows=[]
    for split in ('original','easy','hard'):
        path=folder/(split+'.jsonl')
        if not path.is_file():raise FileNotFoundError(f'Missing dataset: {path}')
        with path.open(encoding='utf-8') as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    counts=Counter(t['question']['type'] for t in rows)
    if len(rows)!=231 or dict(counts)!=EXPECTED or len({t['id'] for t in rows})!=231:
        raise ValueError(f'Expected 231 unique JevBench tasks with {EXPECTED}, found {len(rows)} {dict(counts)}')
    return rows

def subset_tasks(rows, full, per_type, seed=42):
    if full:return rows,'full'
    if not 1<=per_type<=18:raise ValueError('--per-type must be 1..18 (18 Score tasks exist)')
    # Randomly sample each Jev primitive over ALL three splits (not the first
    # N items of the original split), but preserve canonical evaluation order.
    rng=random.Random(seed)
    selected_ids=set()
    for typ in ('choice','noul','score'):
        available=[r for r in rows if r['question']['type']==typ]
        selected_ids.update(r['id'] for r in rng.sample(available,per_type))
    chosen=[r for r in rows if r['id'] in selected_ids]
    assert len(chosen)==3*per_type
    return chosen,f'balanced{per_type}_seed{seed}'

def make_request(task, model):
    from llm2jev import JevRequest,Choice,Score,Noul
    q=task['question'];fields={'instructions':q.get('instructions'),'criteria':q.get('criteria')}
    typ=q['type']
    if typ=='choice':obj=Choice(**fields)
    elif typ=='score':obj=Score(**fields)
    elif typ=='noul':obj=Noul(**fields)
    else:raise ValueError(f'Unknown type {typ}')
    return JevRequest(model=model,state=task['state'],questions={'decision':obj})

def predict(task, req, questions, yes_probabilities, usage):
    from llm2jev.inference.assembler import assemble_response
    answer=assemble_response(request=req,tasks=questions,yes_probabilities=yes_probabilities,usage=usage).answers['decision']
    typ=task['question']['type']
    if typ=='choice':
        predicted=answer.choice
        probs={str(k):float(v) for k,v in answer.probabilities.items()}
    elif typ=='score':
        predicted=max(answer.probabilities,key=answer.probabilities.get)
        probs={str(k):float(v) for k,v in answer.probabilities.items()}
    else:
        predicted='yes' if answer.noul>=.5 else 'no'
        probs={'yes':float(answer.noul),'no':float(1-answer.noul)}
    expected=str(task['expected']);predicted=str(predicted)
    return {'prediction':predicted,'expected':expected,'correct':predicted==expected,
            'probabilities':probs,'confidence':max(probs.values())}

def score_backend(backend, kind, req, questions, prompts, task):
    begin=time.perf_counter()
    outcome={'status':'error','latency_s':None,'candidate_count':len(prompts),
             'candidate_p_yes':None,'predicted':None,'diagnostics':None,'usage':None,'error':None}
    try:
        result=backend.score(model=req.model,prompts=prompts)
        outcome['latency_s']=time.perf_counter()-begin
        values=[float(v) for v in result.yes_probabilities]
        if len(values)!=len(prompts):raise RuntimeError('Candidate count mismatch')
        result_meta=predict(task,req,questions,values,result.usage)
        outcome.update({'status':'ok','candidate_p_yes':values,'predicted':result_meta,
                        'usage':result.usage.to_dict()})
    except Exception as exc:
        outcome['latency_s']=time.perf_counter()-begin
        outcome['error']=f'{type(exc).__name__}: {exc}'
    if kind=='ollama':
        diagnostics=backend.get_diagnostics()
        outcome['diagnostics']=diagnostics
        if outcome['status']=='error' and 'coverage failure' in str(outcome['error']).lower():
            outcome['status']='coverage_failure'
    return outcome

def jsonl_latest(path):
    saved={}
    if path.is_file():
        with path.open(encoding='utf-8') as f:
            for num,line in enumerate(f,1):
                if not line.strip():continue
                try:r=json.loads(line)
                except json.JSONDecodeError as e:raise ValueError(f'Corrupt JSONL {path}:{num}') from e
                key=(r['id'],r['repetition'])
                saved[key]=r
    return saved

def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def run(args):
    all_tasks=tasks_from_dataset(args.dataset_dir)
    if args.one_task:
        if args.full:raise ValueError("--one-task and --full cannot be combined")
        tasks=[all_tasks[0]]
        subset="one_"+safe(tasks[0]["id"])
    else:
        tasks,subset=subset_tasks(all_tasks,args.full,args.per_type,args.seed)
    dataset_sha=sha_json(all_tasks)
    pair_name=safe(args.ollama_model)
    if args.ollama_only:pair_name+='-ollamaonly'
    raw_dir=args.out_dir/'raw';raw_dir.mkdir(parents=True,exist_ok=True)
    path=raw_dir/f'e4_{pair_name}_{subset}_r{args.repetitions}.jsonl'

    # Preflight the Ollama API before loading the HF model into GPU memory.
    from ollama import Client
    from llm2jev import OllamaBackend
    import ollama as ollama_module
    client=Client(host=args.host,timeout=args.timeout)
    with urlopen(args.host.rstrip('/') + '/api/version',timeout=10) as response:
        version=json.loads(response.read().decode('utf-8')).get('version','unknown')
    tags=client.list()
    installed={str(getattr(x,'model',None) or getattr(x,'name',None)):x for x in tags.models}
    if args.ollama_model not in installed:
        raise RuntimeError(f'Model tag {args.ollama_model!r} missing from local ollama list')
    digest=installed[args.ollama_model].digest
    try:
        show=client.show(args.ollama_model)
        ollama_show={'template':getattr(show,'template',None),'details':getattr(show,'details',None).model_dump() if getattr(show,'details',None) else None,
                     'model_info':getattr(show,'model_info',None)}
    except Exception as ex:ollama_show={'show_error':str(ex)}

    try:sha=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,timeout=5).stdout.strip()
    except Exception:sha=None
    hf_path=str(args.hf_model.expanduser().resolve()) if args.hf_model else None
    hf_identity={}
    if args.hf_model and args.hf_model.is_dir():
        for f in ('config.json','tokenizer_config.json','model.safetensors.index.json'):
            item=args.hf_model/f
            if item.is_file():hf_identity[f]=hashlib.sha256(item.read_bytes()).hexdigest()
        hf_identity['weight_file_sizes']={f.name:f.stat().st_size for f in sorted(args.hf_model.glob('*.safetensors'))}
    fingerprint={'protocol':PROTOCOL,'ollama_model':args.ollama_model,'ollama_digest':digest,
                 'hf_model_path':hf_path,'hf_local_metadata':hf_identity,'hf_dtype':args.hf_dtype,'hf_device':args.device,
                 'hf_batch_size':1,'hf_device_map_arg':args.device_map,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'dataset_sha256':dataset_sha,'per_type':args.per_type,
                 'full':args.full,'seed':args.seed,'repetitions':args.repetitions,'top_logprobs':20,
                 'think':False,'temperature':0.,'ollama_only':args.ollama_only,'keep_alive':args.keep_alive}
    cfg_hash=sha_json(fingerprint)
    saved=jsonl_latest(path)
    for k,r in saved.items():
        if r.get('config_sha256')!=cfg_hash:
            raise RuntimeError(f'Resumed output has changed model/dataset/config: {path}; use new --out-dir')
        if k[0] not in {t['id'] for t in tasks} or k[1] not in range(args.repetitions):
            raise RuntimeError('Output contains task/repetition outside requested subset')
    metadata={'fingerprint':fingerprint,'config_sha256':cfg_hash,'python':platform.python_version(),
              'platform':platform.platform(),'ollama_version':version,'ollama_python_package':getattr(ollama_module,'__version__','unknown'),
              'ollama_show':ollama_show,'repo_commit':sha,'notes':[
                  'Pipelines share LLM2Jev binary candidate prompts and assembler.',
                  'They may NOT share identical quantization, underlying model tensors or rendered chat-template tokens.',
                  'HF uses full-vocabulary prefill logits; Ollama uses one generated token, top-20 logprobs.',
                  'HF batch_size=1; serialized execution order alternates by task and repetition.',
                  'Timing is sequential per typed JevBench decision; model initialization excluded.'
              ]}
    meta_file=args.out_dir/'metadata'/f'{pair_name}_{subset}_r{args.repetitions}.json'
    meta_file.parent.mkdir(parents=True,exist_ok=True)
    if not meta_file.exists():meta_file.write_text(json.dumps(metadata,indent=2,ensure_ascii=False,default=str)+'\n')
    print(f'[E4] Ollama {version}, {args.ollama_model}, digest {str(digest)[:16]}',flush=True)
    print(f'[E4] {len(tasks)} tasks, {args.repetitions} repetitions, {len(saved)} already saved',flush=True)
    if args.ollama_only:print('WARNING: Ollama-only mode measures latency/coverage, NOT exact-logit fidelity.',flush=True)
    else:print('WARNING: Cross-stack comparison, not guaranteed identical weights or tokenized chat prompt.',flush=True)

    ollama_backend=OllamaBackend(host=args.host,top_logprobs=20,temperature=0.,think=False,keep_alive=args.keep_alive)
    ollama_backend.client=Client(host=args.host,timeout=args.timeout)
    hf_backend=None
    if not args.ollama_only:
        if not args.hf_model or not args.hf_model.is_dir():
            raise FileNotFoundError(f'--hf-model must point to a locally downloaded HF directory: {args.hf_model}')
        from llm2jev.backend.transformers.backend import TransformersBackend
        print('[E4] Loading full-logit HF reference; this may take time...',flush=True)
        load_start=time.perf_counter()
        hf_backend=TransformersBackend(args.hf_model,device=args.device,device_map=args.device_map,dtype=args.hf_dtype,batch_size=1,enable_thinking=False)
        metadata["hf_device_map_actual"] = getattr(hf_backend.model, "hf_device_map", None)
        if args.device_map:
            print("[E4] HF layer/device allocation:", metadata["hf_device_map_actual"], flush=True)
            used = {str(v) for v in (metadata["hf_device_map_actual"] or {}).values()}
            if any(v in {"cpu", "disk"} for v in used):
                raise RuntimeError("HF checkpoint has CPU/disk offloading; aborting GPU-only latency test")
        print(f'[E4] HF loaded in {time.perf_counter()-load_start:.1f}s',flush=True)
        metadata['hf_tokenization']={'yes_token_id':hf_backend.yes_token_id,
             'no_token_id':hf_backend.no_token_id,
             'yes_decoded':hf_backend.tokenizer.decode([hf_backend.yes_token_id]),
             'no_decoded':hf_backend.tokenizer.decode([hf_backend.no_token_id]),
             'tokenizer_chat_template_sha256':sha_json(hf_backend.tokenizer.chat_template),
             'model_config_class':type(hf_backend.model.config).__name__}
        meta_file.write_text(json.dumps(metadata,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')

    from llm2jev.inference.binary import compile_binary_questions
    from llm2jev.inference.prompt import DefaultPromptRenderer
    renderer=DefaultPromptRenderer()
    try:
        # Do not include initial model load / first-cache compilation in timed measurements.
        first=tasks[0];req=make_request(first,args.ollama_model)
        questions=compile_binary_questions(req);prompts=tuple(renderer.render(q) for q in questions)
        for name,backend in [('ollama',ollama_backend),('transformers',hf_backend)]:
            if backend is None:continue
            print(f'[E4] Warmup: {name} ...',flush=True)
            warm=score_backend(backend,name,req,questions,prompts,first)
            if warm['status']!='ok':
                raise RuntimeError(f'Warmup failed on {name}: {warm["error"]}')
        for ix,task in enumerate(tasks):
            req=make_request(task,args.ollama_model)
            questions=compile_binary_questions(req)
            prompts=tuple(renderer.render(q) for q in questions)
            prompt_sha=sha_json(prompts)
            for rep in range(args.repetitions):
                key=(task['id'],rep)
                old=saved.get(key)
                if old and (not args.retry_errors or all(
                    old.get(name,{}).get('status')=='ok' for name in (('ollama',) if args.ollama_only else ('ollama','transformers')))):
                    continue
                order=['ollama'] if args.ollama_only else (['ollama','transformers'] if (ix+rep)%2==0 else ['transformers','ollama'])
                output={}
                for name in order:
                    backend=ollama_backend if name=='ollama' else hf_backend
                    output[name]=score_backend(backend,name,req,questions,prompts,task)
                pair=None
                a=output.get('ollama',{});b=output.get('transformers',{})
                if a.get('status')=='ok' and b.get('status')=='ok':
                    aa,bb=a['candidate_p_yes'],b['candidate_p_yes']
                    differences=[abs(u-v) for u,v in zip(aa,bb)]
                    if len(differences)!=len(prompts):raise RuntimeError('Pairwise candidate alignment failure')
                    ollama_prompt_tokens=sum(d.get('prompt_tokens',0) for d in a.get('diagnostics') or [])
                    hf_prompt_tokens=(b.get('usage') or {}).get('input_tokens')
                    pair={'ollama_prompt_tokens':ollama_prompt_tokens,
                          'hf_prompt_tokens':hf_prompt_tokens,
                          'prompt_token_count_match':ollama_prompt_tokens==hf_prompt_tokens,
                          'candidate_mae':statistics.mean(differences),'candidate_max_abs_error':max(differences),
                          'typed_prediction_agreement':a['predicted']['prediction']==b['predicted']['prediction'],
                          'typed_probability_l1':sum(abs(a['predicted']['probabilities'][k]-b['predicted']['probabilities'][k]) for k in a['predicted']['probabilities'])}
                row={'protocol':PROTOCOL,'config_sha256':cfg_hash,'id':task['id'],'repetition':rep,
                     'question_type':task['question']['type'],'family':task.get('family'),
                     'gold':str(task['expected']),'prompt_sha256':prompt_sha,'binary_calls':len(prompts),
                     'order':order,'ollama':a,'transformers':b or None,'comparison':pair,
                     'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
                append(path,row);saved[key]=row
                sa=a.get('status','missing');sb=b.get('status','n/a')
                print(f"[{ix+1:3}/{len(tasks)} r{rep+1}] {task['id']} {len(prompts)} binary calls: Ollama={sa} ({a.get('latency_s',0):.2f}s), HF={sb}"+ (f", MAE={pair['candidate_mae']:.4f}" if pair else ''),flush=True)
        print('[E4] Raw results:',path,flush=True)
    finally:
        if hf_backend is not None:hf_backend.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ollama-model',default='qwen3.5:0.8b')
    ap.add_argument('--hf-model',type=Path,help='Local HF model checkpoint directory, required for paired experiment')
    ap.add_argument('--ollama-only',action='store_true',help='Fallback: only Ollama stability/timing, NOT cross-backend fidelity')
    ap.add_argument('--hf-dtype',default='auto',choices=['auto','float16','bfloat16','float32'])
    ap.add_argument('--device-map',choices=['auto','balanced','balanced_low_0','sequential'],default=None,help='Shard HF model across visible CUDA GPUs')
    ap.add_argument('--one-task',action='store_true',help='Run the same first JevBench item only (exactly one task)')
    ap.add_argument('--device',default='cuda',help='HF device (e.g. cuda or cpu)')
    ap.add_argument('--host',default='http://127.0.0.1:11434')
    ap.add_argument('--keep-alive',default='10m')
    ap.add_argument('--timeout',type=float,default=180.,help='Ollama request timeout in seconds (default 180)')
    ap.add_argument('--dataset-dir',type=Path,default=DEFAULT_DATASET)
    ap.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--per-type',type=int,default=5,help='Balanced tasks per Choice/Noul/Score, range 1..18')
    ap.add_argument('--full',action='store_true',help='Run all 231 public JevBench tasks')
    ap.add_argument('--seed',type=int,default=42,help='Reproducible per-type subset sampling seed (default 42)')
    ap.add_argument('--repetitions',type=int,default=3)
    ap.add_argument('--retry-errors',action='store_true')
    args=ap.parse_args()
    if args.repetitions<1:ap.error('--repetitions must be >= 1')
    if args.timeout<=0:ap.error('--timeout must be >0')
    if not args.ollama_only and args.hf_model is None:ap.error('Pass --hf-model /absolute/path or --ollama-only for timing-only')
    run(args)
if __name__=='__main__':main()
