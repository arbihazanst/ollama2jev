#!/usr/bin/env python3
"""Enable E4 one-task testing and optional multi-GPU HF model sharding.

Run from an Ollama2Jev repository or pass --root /path/to/LLM2Jev.
Each change is anchored to exact existing code; originals get .e4_pre_multigpu.bak backups.
Re-running this installer is safe. It does not change archived E4 outputs.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path


def edit_exact(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    count=text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly one matching anchor, found {count}; no files changed')
    return text.replace(old, new, 1)


def transform_backend(s: str) -> str:
    s=edit_exact(s,
        '        device: str | None = None,\n        dtype: str = "auto",',
        '        device: str | None = None,\n        device_map: str | None = None,\n        dtype: str = "auto",',
        'backend signature')
    s=edit_exact(s,
        '        self.model = model_class.from_pretrained(\n            self.model_path,\n            **model_kwargs,\n        )\n        self.model.to(selected_device)\n        self.model.eval()\n        self.device = selected_device',
        '        # Multi-GPU dispatch: do not call .to() on a sharded model.\n'
        '        if device_map:\n'
        '            model_kwargs["device_map"] = device_map\n'
        '        self.model = model_class.from_pretrained(\n'
        '            self.model_path,\n'
        '            **model_kwargs,\n'
        '        )\n'
        '        if device_map:\n'
        '            embedding = self.model.get_input_embeddings()\n'
        '            self.device = embedding.weight.device if embedding is not None else next(self.model.parameters()).device\n'
        '        else:\n'
        '            self.model.to(selected_device)\n'
        '            self.device = selected_device\n'
        '        self.model.eval()',
        'backend model loading')
    return s


def transform_runner(s: str) -> str:
    s=edit_exact(s,
        '    tasks,subset=subset_tasks(all_tasks,args.full,args.per_type,args.seed)',
        '    if args.one_task:\n'
        '        if args.full:raise ValueError("--one-task and --full cannot be combined")\n'
        '        tasks=[all_tasks[0]]\n'
        '        subset="one_"+safe(tasks[0]["id"])\n'
        '    else:\n'
        '        tasks,subset=subset_tasks(all_tasks,args.full,args.per_type,args.seed)',
        'runner one-task selection')
    s=edit_exact(s,
        "'hf_batch_size':1,'dataset_sha256':dataset_sha,",
        "'hf_batch_size':1,'hf_device_map_arg':args.device_map,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'dataset_sha256':dataset_sha,",
        'runner fingerprint')
    s=edit_exact(s,
        'hf_backend=TransformersBackend(args.hf_model,device=args.device,dtype=args.hf_dtype,batch_size=1,enable_thinking=False)',
        'hf_backend=TransformersBackend(args.hf_model,device=args.device,device_map=args.device_map,dtype=args.hf_dtype,batch_size=1,enable_thinking=False)\n'
        '        metadata["hf_device_map_actual"] = getattr(hf_backend.model, "hf_device_map", None)\n'
        '        if args.device_map:\n'
        '            print("[E4] HF layer/device allocation:", metadata["hf_device_map_actual"], flush=True)\n'
        '            used = {str(v) for v in (metadata["hf_device_map_actual"] or {}).values()}\n'
        '            if any(v in {"cpu", "disk"} for v in used):\n'
        '                raise RuntimeError("HF checkpoint has CPU/disk offloading; aborting GPU-only latency test")',
        'runner model creation')
    s=edit_exact(s,
        "    ap.add_argument('--hf-dtype',default='auto',choices=['auto','float16','bfloat16','float32'])",
        "    ap.add_argument('--hf-dtype',default='auto',choices=['auto','float16','bfloat16','float32'])\n"
        "    ap.add_argument('--device-map',choices=['auto','balanced','balanced_low_0','sequential'],default=None,help='Shard HF model across visible CUDA GPUs')\n"
        "    ap.add_argument('--one-task',action='store_true',help='Run the same first JevBench item only (exactly one task)')",
        'runner CLI flags')
    return s


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    a=ap.parse_args()
    files={
       a.root/'src/llm2jev/backend/transformers/backend.py':transform_backend,
       a.root/'experiments/04_backend_fidelity/scripts/run_e4_paired.py':transform_runner,
    }
    changes=[]
    for path,transform in files.items():
        if not path.is_file():raise SystemExit(f'Cannot find {path}. Pass --root if not in repo root.')
        before=path.read_text(encoding='utf-8')
        after=transform(before)
        ast.parse(after,filename=str(path))
        changes.append((path,before,after))
    for path,before,after in changes:
        if before==after:
            print('Already patched:',path)
            continue
        backup=path.with_name(path.name+'.e4_pre_multigpu.bak')
        if not backup.exists():backup.write_text(before,encoding='utf-8')
        path.write_text(after,encoding='utf-8')
        print('Patched:',path)
    print('Ready. First test with --one-task --repetitions 1 --device-map auto.')

if __name__=='__main__':main()
