#!/usr/bin/env python3
"""E3: native Ollama /v1/systemone decision-model evaluation on public JevBench.

One request per task; strict full-denominator accuracy; append-only resumable JSONL.
Uses only Python standard library. Run from any working directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODELS = (
    "tev1:0.8b", "tev1:4b", "nimble-v3:9b",
    "clef-flash:9b", "clef:27b", "pplx-decider:latest",
)
PROTOCOL = "e3-native-systemone-v1"
TYPES = {"choice": 139, "noul": 74, "score": 18}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = ROOT.parent / "jevbench" / "datasets" / "public"
DEFAULT_OUT = ROOT / "experiments" / "03_finetuned_decision_models" / "results"


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def sha_json(data):
    data_str = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def safe_model(model):
    return model.replace(":", "-").replace("/", "-")


def http_json(base_url, route, payload=None, timeout=120):
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(base_url.rstrip("/") + route, data=body, headers=headers,
                  method="POST" if body is not None else "GET")
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as ex:
        snippet = ex.read(1200).decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {ex.code}: {snippet}") from ex
    except (URLError, TimeoutError) as ex:
        raise RuntimeError(f"Ollama network/timeout failure: {ex}") from ex


def load_tasks(dataset_dir, smoke=False):
    rows = []
    for split in ("original", "easy", "hard"):
        path = dataset_dir / (split + ".jsonl")
        if not path.is_file():
            raise FileNotFoundError(f"Missing JevBench dataset: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    counts = Counter(t["question"]["type"] for t in rows)
    if len(rows) != 231 or dict(counts) != TYPES:
        raise ValueError(f"Wrong JevBench composition: {len(rows)} items; {dict(counts)}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate dataset item IDs")
    for t in rows:
        q = t["question"]
        if q["type"] == "choice" and len(q["criteria"]) < 2:
            raise ValueError("Choice requires two or more options")
        if q["type"] == "score" and not (2 <= len(q["criteria"]) <= 10):
            raise ValueError("Score requires 2--10 levels")
    if smoke:
        return rows[:1]

    return rows


def request_body(task, model, keep_alive):
    # Retain question fields exactly as recorded, including descriptions and order.
    spec = task["question"]
    q = {"type": spec["type"], "instructions": spec.get("instructions")}
    if "criteria" in spec and spec["criteria"] is not None:
        q["criteria"] = spec["criteria"]
    return {"model": model, "state": task["state"],
            "questions": {"decision": q}, "keep_alive": keep_alive}


def distribution_from_answer(task, answer):
    """Return expected, predicted, normalized label probabilities (ordered).

    Choice uses native 'choice' as prediction but validates agreement with argmax.
    Score uses argmax over per-level probabilities (same as original JevBench runner).
    Noul uses P(true) >= 0.5 to predict yes, as E1 does.
    """
    if not isinstance(answer, dict):
        raise ValueError("Missing or non-object decision answer")
    kind = task["question"]["type"]
    if kind == "noul":
        prob = answer.get("noul")
        if isinstance(prob, bool) or not isinstance(prob, (float, int)) or not math.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("Missing/invalid noul probability")
        prob = float(prob)
        probs = {"yes": prob, "no": 1 - prob}
        pred = "yes" if prob >= 0.5 else "no"
        expected = task["expected"]
        if expected not in ("yes", "no"):
            raise ValueError("Unexpected Noul gold label")
        return expected, pred, probs

    if kind == "choice":
        labels = list(task["question"]["criteria"].keys())
        pred = answer.get("choice")
        if pred not in labels:
            raise ValueError("Native choice not in allowed labels")
    else:
        labels = [str(i) for i in range(len(task["question"]["criteria"]))]

    supplied = answer.get("probabilities")
    if not isinstance(supplied, dict) or set(supplied) != set(labels):
        raise ValueError("Probability keys do not match candidate set")
    probs = {}
    for label in labels:
        val = supplied[label]
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or not 0 <= val <= 1:
            raise ValueError(f"Invalid probability: {label}")
        probs[label] = float(val)
    total = sum(probs.values())
    if not 0.98 <= total <= 1.02:
        raise ValueError(f"Probabilities do not sum to approximately 1: {total}")
    # Small output rounding is normalized for proper scoring.
    probs = {key: val / total for key, val in probs.items()}
    maximum = max(probs.values())
    argmax = next(label for label in labels if abs(probs[label] - maximum) < 1e-12)
    if kind == "choice":
        # Native ties are accepted; they can follow implementation-specific order.
        if abs(probs[pred] - maximum) > 1e-8:
            raise ValueError(f"Reported choice {pred!r} disagrees with maximum probability {argmax!r}")
        expected = task["expected"]
        if expected not in labels:
            raise ValueError("Choice gold label not in criteria")
        return expected, pred, probs
    expected = str(int(task["expected"]))
    if expected not in labels:
        raise ValueError("Score gold level outside rubric")
    return expected, argmax, probs


def probability_metrics(pred, expected, probs):
    if expected not in probs:
        raise ValueError("Gold class absent from probabilities")
    brier = sum((v - (1.0 if label == expected else 0.0)) ** 2
                for label, v in probs.items())
    nll = -math.log(max(probs[expected], 1e-15))
    return brier, nll, max(probs.values())


def load_latest(path, allowed_ids=None):
    by_id = {}
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as ex:
                    raise ValueError(f"Corrupt JSONL at {path}:{number}; repair before resuming") from ex
                if row.get("protocol") != PROTOCOL:
                    raise ValueError(f"Incompatible protocol at {path}:{number}")
                if allowed_ids is not None and row.get("id") not in allowed_ids:
                    raise ValueError(f"Unknown benchmark item {row.get('id')!r} in {path}")
                by_id[row["id"]] = row
    return by_id


def append_result(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ece10(rows):
    if not rows:
        return None
    n = len(rows)
    total = 0.0
    for b in range(10):
        selected = [r for r in rows if min(9, int(r["confidence_for_calibration"] * 10)) == b]
        if selected:
            accuracy = sum(bool(r["correct"]) for r in selected) / len(selected)
            confidence = sum(r["confidence_for_calibration"] for r in selected) / len(selected)
            total += (len(selected) / n) * abs(accuracy - confidence)
    return total


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    spot = (len(ordered) - 1) * p
    low, high = math.floor(spot), math.ceil(spot)
    return ordered[low] + (ordered[high] - ordered[low]) * (spot - low)


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    proportion = k / n
    denom = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denom
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * n)) / n) / denom
    return [max(0, center - margin), min(1, center + margin)]


def aggregate_model(model, tasks, result_root, smoke):
    path = result_root / "raw" / f"e3_jevbench_{safe_model(model)}_full.jsonl"
    latest = load_latest(path, {t["id"] for t in tasks})
    records = [latest[t["id"]] for t in tasks if t["id"] in latest]
    valid = [r for r in records if r.get("status") == "ok"]
    correct = sum(bool(r.get("correct")) for r in records)
    by_type = {}
    for typ in ("choice", "noul", "score"):
        typ_rows = [r for r in records if r["question_type"] == typ]
        denom = sum(t["question"]["type"] == typ for t in tasks)
        by_type[typ] = {"tasks": denom, "recorded": len(typ_rows),
                        "successful": sum(r["status"] == "ok" for r in typ_rows),
                        "correct": sum(bool(r.get("correct")) for r in typ_rows),
                        "accuracy": sum(bool(r.get("correct")) for r in typ_rows) / denom}
    latencies = [r["latency_s"] for r in valid]
    warm = latencies[1:]
    result = {"model": model, "protocol": PROTOCOL, "mode": "smoke" if smoke else "full",
              "tasks": len(tasks), "recorded": len(records), "successful": len(valid),
              "invalid": sum(r["status"] == "invalid_response" for r in records),
              "errors": sum(r["status"] == "error" for r in records),
              "correct": correct, "accuracy": correct / len(tasks),
              "ci_wilson_95": wilson(correct, len(tasks)),
              "by_type": by_type,
              "probability_valid_n": len(valid),
              "brier": statistics.mean(r["brier"] for r in valid) if valid else None,
              "ece10": ece10(valid),
              "nll": statistics.mean(r["nll"] for r in valid) if valid else None,
              "first_success_latency_s": latencies[0] if latencies else None,
              "warm_p50_s": percentile(warm, 0.5),
              "warm_p95_s": percentile(warm, 0.95),
              "mean_input_tokens_per_success": statistics.mean(r["input_tokens"] for r in valid
                    if isinstance(r.get("input_tokens"), int)) if any(isinstance(r.get("input_tokens"), int) for r in valid) else None,
              "mean_output_tokens_per_success": statistics.mean(r["output_tokens"] for r in valid
                    if isinstance(r.get("output_tokens"), int)) if any(isinstance(r.get("output_tokens"), int) for r in valid) else None}
    return result


def version_tuple(version):
    nums = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
    return tuple(map(int, nums.groups())) if nums else None


def preflight(base_url, models, result_root, timeout):
    version = http_json(base_url, "/api/version", timeout=10).get("version", "unknown")
    print(f"Ollama version: {version}", flush=True)
    if "nimble-v3:9b" in models:
        parsed = version_tuple(version)
        if parsed is None or parsed < (0, 40, 4):
            raise RuntimeError("nimble-v3:9b needs Ollama >= 0.40.4; upgrade before E3")
    if any(m.startswith(("clef", "pplx-decider")) for m in models):
        parsed = version_tuple(version)
        if parsed is not None and parsed < (0, 35, 1):
            raise RuntimeError("Clef and other newer decision models require upgraded Ollama")
    data = http_json(base_url, "/api/tags", timeout=10)
    installed = {m.get("name") or m.get("model"): m for m in data.get("models", [])}
    missing = [m for m in models if m not in installed]
    if missing:
        raise RuntimeError("Models not installed with these exact tags: " + ", ".join(missing)
                           + "\nCheck `ollama list` and `ollama pull MODEL`; do not silently substitute an alias.")
    result_root.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        git_sha = None
    for model in models:
        meta = {"captured_at": now_iso(), "model": model, "digest": installed[model].get("digest"),
                "size": installed[model].get("size"), "modified_at": installed[model].get("modified_at"),
                "ollama_version": version, "python": platform.python_version(),
                "platform": platform.platform(), "repo_git_sha": git_sha}
        try:
            show = http_json(base_url, "/api/show", {"model": model}, timeout=min(timeout, 30))
            meta["model_info"] = show.get("model_info", {})
            meta["details"] = show.get("details", {})
            meta["capabilities"] = show.get("capabilities", [])
        except Exception as ex:
            meta["show_error"] = str(ex)
        (result_root / "metadata" / (safe_model(model) + ".json")).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Model available: {model}, digest={str(meta['digest'])[:16]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=1,
                        help="Number of retries after initial HTTP failure")
    parser.add_argument("--keep-alive", default="10m")
    parser.add_argument("--smoke", action="store_true", help="5 tasks per primitive; separate result directory")
    parser.add_argument("--retry-errors", action="store_true", help="Retry previously recorded HTTP errors")
    parser.add_argument("--retry-invalid", action="store_true", help="Retry previously invalid-format responses")
    parser.add_argument("--summary-only", action="store_true", help="Recompute summaries without Ollama")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after first error; useful for smoke tests")
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 0:
        parser.error("--timeout must be >0 and --retries >=0")
    tasks = load_tasks(args.dataset_dir, args.smoke)
    result_root = args.out_dir / "smoke" if args.smoke else args.out_dir
    for model in args.models:
        path = result_root / "raw" / f"e3_jevbench_{safe_model(model)}_full.jsonl"
        saved = load_latest(path, {t["id"] for t in tasks})
        # Disallow silently reusing stale results created with a different dataset.
        for task in tasks:
            if task["id"] in saved:
                expected_hash = sha_json(request_body(task, model, args.keep_alive))
                if saved[task["id"]].get("request_sha256") != expected_hash:
                    raise RuntimeError(f"Request hash mismatch for {model}/{task['id']}; "
                                       "use the same --keep-alive and dataset, or a fresh --out-dir")
    if not args.summary_only:
        preflight(args.host, args.models, result_root, args.timeout)
    for model in args.models:
        path = result_root / "raw" / f"e3_jevbench_{safe_model(model)}_full.jsonl"
        seen = load_latest(path, {t["id"] for t in tasks})
        print("\n" + "=" * 72 + f"\nMODEL {model}: recorded {len(seen)}/{len(tasks)}\n" + "=" * 72, flush=True)
        if args.summary_only:
            continue
        for index, task in enumerate(tasks, 1):
            old = seen.get(task["id"])
            if old and ((old["status"] == "ok") or
                        (old["status"] == "error" and not args.retry_errors) or
                        (old["status"] == "invalid_response" and not args.retry_invalid)):
                continue
            body = request_body(task, model, args.keep_alive)
            row = {"id": task["id"], "model": model, "protocol": PROTOCOL,
                   "question_type": task["question"]["type"], "family": task.get("family"),
                   "split": task.get("split"), "expected": task["expected"],
                   "request_sha256": sha_json(body), "timestamp": now_iso(),
                   "status": None, "correct": False, "latency_s": None,
                   "prediction": None, "probabilities": None,
                   "raw_answer": None, "raw_response": None, "error": None,
                   "input_tokens": None, "output_tokens": None,
                   "brier": None, "ece_confidence": None, "nll": None}
            t0 = time.perf_counter()
            response = None
            for attempt in range(args.retries + 1):
                try:
                    response = http_json(args.host, "/v1/systemone", body, timeout=args.timeout)
                    break
                except Exception as ex:
                    row["error"] = str(ex)
                    print(f"[{model}] {index}/{len(tasks)} attempt {attempt + 1}: {str(ex)[:160]}", flush=True)
                    if attempt < args.retries:
                        time.sleep(min(3 * (attempt + 1), 15))
            row["latency_s"] = time.perf_counter() - t0
            if response is None:
                row["status"] = "error"
            else:
                row["raw_response"] = response
                usage = response.get("usage") or {}
                row["input_tokens"] = usage.get("input_tokens")
                row["output_tokens"] = usage.get("output_tokens")
                answer = (response.get("answers") or {}).get("decision")
                row["raw_answer"] = answer
                try:
                    expected, pred, probabilities = distribution_from_answer(task, answer)
                    row["status"] = "ok"
                    row["prediction"] = int(pred) if task["question"]["type"] == "score" else pred
                    row["correct"] = pred == expected
                    row["probabilities"] = probabilities
                    brier, nll, conf = probability_metrics(pred, expected, probabilities)
                    row["brier"], row["nll"] = brier, nll
                    row["confidence_for_calibration"] = conf
                    row["ece_confidence"] = conf
                    row["error"] = None
                except Exception as ex:
                    row["status"] = "invalid_response"
                    row["error"] = f"{type(ex).__name__}: {ex}"
            append_result(path, row)
            seen[task["id"]] = row
            print(f"[{model}] {index:3d}/{len(tasks)} {task['id']}: "
                  f"{row['status']} ({'CORRECT' if row['correct'] else 'WRONG'}), "
                  f"prediction={row['prediction']!r}, {row['latency_s']:.3f}s", flush=True)
            if row["status"] != "ok" and args.stop_on_error:
                print("Stopping after error; inspect saved JSONL before continuing.", flush=True)
                break
        outcome = aggregate_model(model, tasks, result_root, args.smoke)
        print(f"[SUMMARY] {model}: {outcome['correct']}/{outcome['tasks']} "
              f"({outcome['accuracy']:.3f}), successful={outcome['successful']}, "
              f"invalid={outcome['invalid']}, errors={outcome['errors']}", flush=True)
    # Recompute summaries from recorded data rather than in-memory state.
    summaries = [aggregate_model(model, tasks, result_root, args.smoke) for model in MODELS
                 if (result_root / "raw" / f"e3_jevbench_{safe_model(model)}_full.jsonl").is_file()]
    out = result_root / "summary" / "e3_model_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved summary: {out}", flush=True)
    if any(r["recorded"] != r["tasks"] for r in summaries):
        print("WARNING: Some models are incomplete; do not report final accuracy comparisons yet.", flush=True)


if __name__ == "__main__":
    main()
