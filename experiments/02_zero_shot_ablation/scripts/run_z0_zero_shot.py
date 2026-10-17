#!/usr/bin/env python3
"""Z0: conventional direct zero-shot Ollama generation for public JevBench.

No LLM2Jev binary scoring, logprobs, forced JSON format, few-shot examples,
or answer-repair are used. One Ollama /api/chat request per typed decision.

Run from any working directory; defaults assume sibling repos:
  <parent>/LLM2Jev/
  <parent>/jevbench/datasets/public/{original,easy,hard}.jsonl

This program writes append-only JSONL and resumes model/task pairs after
interrupted SSH/tmux runs. It treats invalid model replies as incorrect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from ollama import Client
except ImportError:
    print("Missing 'ollama' Python package. Install with: python -m pip install ollama", file=sys.stderr)
    raise

MODELS = [
    "qwen3.5:0.8b",
    "qwen3.5:2b",
    "qwen3.5:4b",
    "qwen3.5:9b",
    "qwen3.8:latest",
    "llama3.2:1b",
    "llama3.2:3b",
    "granite4.2:3b",
    "granite4.2:8b",
    "gemma4:e2b",
    "gemma4:e4b",
]

PROMPT_VERSION = "z0-direct-v1"
EXPECTED_TYPES = {"choice": 139, "noul": 74, "score": 18}
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT.parent / "jevbench" / "datasets" / "public"
DEFAULT_RESULT_DIR = REPO_ROOT / "experiments" / "02_zero_shot_ablation" / "results" / "z0_raw"


def as_text(value: Any) -> str:
    """Preserve structured state/instructions as readable source content."""
    if value is None:
        return "(none)"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def load_tasks(dataset_dir: Path, smoke: bool) -> list[dict[str, Any]]:
    tasks = []
    for name in ("original", "easy", "hard"):
        path = dataset_dir / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing JevBench file: {path}")
        with path.open(encoding="utf-8") as stream:
            for line_num, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        tasks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON at {path}:{line_num}") from e

    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate JevBench task IDs; cannot resume safely")

    counts = Counter(t["question"]["type"] for t in tasks)
    if len(tasks) != 231 or dict(counts) != EXPECTED_TYPES:
        raise ValueError(
            f"Dataset differs from expected 231-item public benchmark: "
            f"total={len(tasks)}, types={dict(counts)}"
        )

    if smoke:
        picked: list[dict[str, Any]] = []
        selected = Counter()
        for task in tasks:
            question_type = task["question"]["type"]
            if selected[question_type] < 5:
                picked.append(task)
                selected[question_type] += 1
        assert len(picked) == 15
        return picked
    return tasks


def label_set(task: dict[str, Any]) -> list[str]:
    spec = task["question"]
    kind = spec["type"]
    if kind == "choice":
        return list(spec["criteria"].keys())
    if kind == "score":
        return [str(i) for i in range(len(spec["criteria"]))]
    if kind == "noul":
        return ["yes", "no"]
    raise ValueError(f"Unsupported question type: {kind}")


def make_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    spec = task["question"]
    kind = spec["type"]
    instructions = as_text(spec.get("instructions"))
    state = as_text(task["state"])
    criteria = spec.get("criteria")

    if kind == "choice":
        lines = [f"{name}: {as_text(description)}" for name, description in criteria.items()]
        candidates = "\n".join(lines)
        task_instruction = "Select exactly ONE option label from the candidates."
    elif kind == "score":
        lines = [f"{i}: {as_text(description)}" for i, description in enumerate(criteria)]
        candidates = "\n".join(lines)
        task_instruction = "Select exactly ONE integer level index from the candidates."
    elif kind == "noul":
        if criteria:
            candidates = (
                f"yes: {as_text(criteria.get('true'))}\n"
                f"no: {as_text(criteria.get('false'))}"
            )
        else:
            candidates = "yes: The proposition is true.\nno: The proposition is false."
        task_instruction = "Select exactly ONE answer: yes or no."
    else:
        raise ValueError(f"Unsupported question type: {kind}")

    system = (
        "You are solving a structured decision task. Follow the provided "
        "task instructions and state, then output exactly one allowed label. "
        "Do not produce reasoning, explanations, JSON, quotation marks, "
        "markdown, or extra words."
    )
    user = (
        f"Question type: {kind}\n\n"
        f"Instructions:\n{instructions}\n\n"
        f"State / evidence:\n{state}\n\n"
        f"Allowed labels and criteria:\n{candidates}\n\n"
        f"{task_instruction}\n"
        "Respond with only the exact label."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_direct_answer(raw: str, allowed: list[str]) -> str | None:
    """Strict exact-match except surrounding whitespace; no answer extraction."""
    candidate = raw.strip()
    return candidate if candidate in allowed else None


def safe_model_name(model: str) -> str:
    return model.replace(":", "-").replace("/", "-")


def read_existing(path: Path, known_ids: set[str]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Corrupted result JSONL at {path}:{lineno}: {e}") from e
            task_id = record.get("id")
            if task_id not in known_ids:
                raise ValueError(f"Found unexpected task ID in {path}:{lineno}: {task_id}")
            if record.get("prompt_version") != PROMPT_VERSION:
                raise ValueError(f"Prompt version mismatch in {path}:{lineno}")
            latest[task_id] = record
    return latest


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def infer_one(client: Any, task: dict[str, Any], model: str, options: dict[str, Any],
              keep_alive: str) -> dict[str, Any]:
    messages = make_messages(task)
    raw_messages = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    prompt_hash = hashlib.sha256(raw_messages.encode("utf-8")).hexdigest()
    allowed = label_set(task)
    expected = str(task["expected"]) if task["question"]["type"] == "score" else task["expected"]
    record: dict[str, Any] = {
        "id": task["id"],
        "model": model,
        "family": task.get("family"),
        "question_type": task["question"]["type"],
        "expected": expected,
        "allowed_labels": allowed,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash,
        "decoding": {"temperature": options["temperature"],
                     "num_predict": options["num_predict"], "think": False,
                     "seed": options["seed"]},
    }
    started = time.perf_counter()
    try:
        response = client.chat(
            model=model, messages=messages, stream=False, think=False,
            options=options, keep_alive=keep_alive,
        )
        record["latency_s"] = time.perf_counter() - started
        message = get_field(response, "message", {})
        raw = get_field(message, "content", "") or ""
        prediction = parse_direct_answer(raw, allowed)
        record.update({
            "raw_output": raw,
            "prediction": int(prediction) if prediction is not None and record["question_type"] == "score" else prediction,
            "status": "ok" if prediction is not None else "invalid",
            "correct": prediction == str(expected) if prediction is not None else False,
            "input_tokens": get_field(response, "prompt_eval_count"),
            "output_tokens": get_field(response, "eval_count"),
            "total_duration_ns": get_field(response, "total_duration"),
            "prompt_eval_duration_ns": get_field(response, "prompt_eval_duration"),
            "eval_duration_ns": get_field(response, "eval_duration"),
            "done_reason": get_field(response, "done_reason"),
        })
    except Exception as exc:
        record.update({
            "latency_s": time.perf_counter() - started,
            "raw_output": None,
            "prediction": None,
            "status": "error",
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
            "input_tokens": None,
            "output_tokens": None,
        })
    return record


def quantile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return ordered[lo] * (hi - x) + ordered[hi] * (x - lo)


def summarize(model: str, tasks: list[dict[str, Any]], latest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = [latest[t["id"]] for t in tasks if t["id"] in latest]
    n = len(tasks)
    correct = sum(1 for r in ordered if r.get("correct"))
    valid = sum(1 for r in ordered if r.get("status") == "ok")
    invalid = sum(1 for r in ordered if r.get("status") == "invalid")
    errors = sum(1 for r in ordered if r.get("status") == "error")
    by_type = {}
    for kind in ("choice", "noul", "score"):
        category = [t for t in tasks if t["question"]["type"] == kind]
        sub = [latest[t["id"]] for t in category if t["id"] in latest]
        by_type[kind] = {
            "tasks": len(category),
            "completed": len(sub),
            "valid": sum(r.get("status") == "ok" for r in sub),
            "correct": sum(bool(r.get("correct")) for r in sub),
            "accuracy": sum(bool(r.get("correct")) for r in sub) / len(category),
        }
    warm = [float(r["latency_s"]) for r in ordered[1:] if isinstance(r.get("latency_s"), (float, int))]
    output = [float(r["output_tokens"]) for r in ordered if isinstance(r.get("output_tokens"), (float, int))]
    return {
        "model": model, "method": "Z0_direct_zero_shot_generation", "prompt_version": PROMPT_VERSION,
        "tasks": n, "completed": len(ordered), "valid": valid, "invalid": invalid,
        "errors": errors, "correct": correct, "accuracy": correct / n,
        "by_type": by_type,
        "first_decision_latency_s": ordered[0].get("latency_s") if ordered else None,
        "warm_p50_latency_s": quantile(warm, .50),
        "warm_p95_latency_s": quantile(warm, .95),
        "mean_output_tokens_per_completed_decision": statistics.mean(output) if output else None,
    }


def write_summaries(result_dir: Path, models: list[str], tasks: list[dict[str, Any]]) -> None:
    known = {t["id"] for t in tasks}
    summaries = []
    for model in models:
        file = result_dir / f"z0_jevbench_{safe_model_name(model)}_full.jsonl"
        latest = read_existing(file, known)
        summaries.append(summarize(model, tasks, latest))
    dest = result_dir / "z0_model_summary.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, help="Ollama model tags, default all 11")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--smoke", action="store_true", help="Run a separate balanced 15-task sample")
    parser.add_argument("--num-predict", type=int, default=64, help="Maximum tokens for one direct answer")
    parser.add_argument("--timeout", type=float, default=900.0, help="HTTP timeout per Ollama call, seconds")
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--retry-errors", action="store_true", help="Retry previously recorded transport errors")
    args = parser.parse_args()
    if args.num_predict < 1:
        parser.error("--num-predict must be positive")
    tasks = load_tasks(args.dataset_dir, args.smoke)
    result_dir = args.result_dir / "smoke" if args.smoke else args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    options = {"temperature": 0.0, "num_predict": args.num_predict, "seed": 0}
    client = Client(host=args.host, timeout=args.timeout)

    print(f"[Z0] {len(tasks)} tasks; model count={len(args.models)}; prompt={PROMPT_VERSION}", flush=True)
    print(f"[Z0] Results: {result_dir}", flush=True)
    print("[Z0] Zero-shot direct generation, no few-shot examples, no logprob scoring.", flush=True)
    for model in args.models:
        filename = result_dir / f"z0_jevbench_{safe_model_name(model)}_full.jsonl"
        latest = read_existing(filename, {t["id"] for t in tasks})
        print(f"\n{'='*70}\nMODEL: {model}; already recorded: {len(latest)}\n{'='*70}", flush=True)
        with filename.open("a", encoding="utf-8", buffering=1) as stream:
            for i, task in enumerate(tasks, 1):
                previous = latest.get(task["id"])
                if previous is not None and (previous.get("status") != "error" or not args.retry_errors):
                    continue
                rec = infer_one(client, task, model, options, args.keep_alive)
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stream.flush()
                latest[task["id"]] = rec
                outcome = "CORRECT" if rec["correct"] else rec["status"].upper()
                print(f"[{model}] {i:3d}/{len(tasks)} {task['id']}: {outcome}; "
                      f"pred={rec.get('prediction')!r}; {rec['latency_s']:.2f}s", flush=True)
        summary = summarize(model, tasks, latest)
        print(f"[SUMMARY] {model}: {summary['correct']}/{summary['tasks']} "
              f"({summary['accuracy']:.3f}); valid={summary['valid']}; "
              f"invalid={summary['invalid']}; errors={summary['errors']}", flush=True)
        write_summaries(result_dir, args.models, tasks)
    print("[Z0] Done. Summary: " + str(result_dir / "z0_model_summary.json"), flush=True)


if __name__ == "__main__":
    main()
