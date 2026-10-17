#!/usr/bin/env python3
"""Compare native System-One models (E3) against corresponding E1 Ollama2Jev models.

Only emits final six paired McNemar+Holm comparisons when all six E3 and
parent E1 231-item records exist. Missing/invalid tasks are counted wrong.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

PAIRS = (
    ("qwen3.5:0.8b", "tev1:0.8b"),
    ("qwen3.5:4b", "tev1:4b"),
    ("qwen3.5:9b", "nimble-v3:9b"),
    ("qwen3.5:9b", "clef-flash:9b"),
    ("qwen3.8:latest", "clef:27b"),
    ("qwen3.8:latest", "pplx-decider:latest"),
)
ROOT = Path(__file__).resolve().parents[3]
E1_RAW = ROOT / "experiments" / "01_general_llm_benchmark" / "results" / "raw"
E3_RAW = ROOT / "experiments" / "03_finetuned_decision_models" / "results" / "raw"
COMPARISONS = ROOT / "experiments" / "03_finetuned_decision_models" / "results" / "comparisons"
DATASETS = ROOT.parent / "jevbench" / "datasets" / "public"


def tag(tag_name):
    return tag_name.replace(":", "-").replace("/", "-")


def load_unique_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}")
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r  # the latest saved E3 attempt wins
    return rows


def exact_mcnemar(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) / (2 ** n) for i in range(k + 1)))


def get_tasks(dataset_dir):
    tasks = []
    for split in ("original", "easy", "hard"):
        with (dataset_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
            tasks.extend(json.loads(line) for line in stream if line.strip())
    if len(tasks) != 231 or len(set(t["id"] for t in tasks)) != 231:
        raise ValueError("Expected 231 unique public JevBench tasks")
    return {t["id"]: t for t in tasks}


def e1_distribution(row, task):
    if "prediction" not in row:
        return None
    q = [d["p_yes"] for d in row["diagnostics"]]
    if not q or any(v is None for v in q):
        return None
    typ = task["question"]["type"]
    if typ == "noul":
        yes = q[0] / sum(q) if len(q) == 2 and sum(q) else q[0]
        return {"yes": yes, "no": 1 - yes}
    labels = (list(task["question"]["criteria"].keys()) if typ == "choice"
              else [str(i) for i in range(len(task["question"]["criteria"]))])
    if len(labels) != len(q):
        raise ValueError(f"Binary candidate count mismatch: {row['id']}")
    total = sum(q)
    weights = ([v / total for v in q] if total else [1 / len(q)] * len(q))
    return dict(zip(labels, weights))


def get_quality(rows, tasks, native):
    records = []
    for key, row in rows.items():
        if key not in tasks:
            raise ValueError(f"Unknown task ID: {key}")
        if native:
            if row.get("status") != "ok":
                continue
            probs = row.get("probabilities")
        else:
            probs = e1_distribution(row, tasks[key])
        if not probs:
            continue
        gold = str(tasks[key]["expected"])
        if gold not in probs:
            raise ValueError(f"Expected class not in probability distribution: {key}")
        values = list(probs.values())
        if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise ValueError(f"Unnormalized probability distribution: {key}")
        brier = sum((p - float(label == gold))**2 for label, p in probs.items())
        nll = -math.log(max(float(probs[gold]), 1e-15))
        records.append((brier, nll, max(values), int(bool(row["correct"]))))
    if not records:
        return {"valid": 0, "brier": None, "nll": None, "ece10": None}
    ece = 0.0
    for bin_idx in range(10):
        selected = [r for r in records if min(9, int(r[2] * 10)) == bin_idx]
        if selected:
            ece += (len(selected) / len(records)) * abs(
                statistics.mean(r[3] for r in selected) - statistics.mean(r[2] for r in selected))
    return {"valid": len(records), "brier": statistics.mean(r[0] for r in records),
            "nll": statistics.mean(r[1] for r in records), "ece10": ece}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-raw", type=Path, default=E1_RAW)
    parser.add_argument("--e3-raw", type=Path, default=E3_RAW)
    parser.add_argument("--out-dir", type=Path, default=COMPARISONS)
    parser.add_argument("--dataset-dir", type=Path, default=DATASETS)
    args = parser.parse_args()
    tasks = get_tasks(args.dataset_dir)
    rows = []
    for parent, specialized in PAIRS:
        a_path = args.e1_raw / f"ollama2jev_jevbench_{tag(parent)}_full.jsonl"
        b_path = args.e3_raw / f"e3_jevbench_{tag(specialized)}_full.jsonl"
        a = load_unique_jsonl(a_path)
        b = load_unique_jsonl(b_path)
        if len(a) != 231 or len(b) != 231 or set(a) != set(b):
            raise RuntimeError(f"Incomplete/mismatched 231 JevBench task IDs for {parent} versus {specialized}: "
                               f"{len(a)} parent, {len(b)} specialized; refusing significance calculation")
        a_only = b_only = a_correct = b_correct = 0
        for key in a:
            x, y = bool(a[key].get("correct")), bool(b[key].get("correct"))
            a_correct += x
            b_correct += y
            if x and not y:
                a_only += 1
            elif y and not x:
                b_only += 1
        parent_quality = get_quality(a, tasks, native=False)
        decision_quality = get_quality(b, tasks, native=True)
        rows.append({
            "parent": parent, "specialized": specialized,
            "parent_correct": a_correct, "parent_accuracy": a_correct / 231,
            "specialized_correct": b_correct, "specialized_accuracy": b_correct / 231,
            "delta_accuracy_pp": 100 * (b_correct - a_correct) / 231,
            "parent_only": a_only, "specialized_only": b_only,
            "exact_p": exact_mcnemar(a_only, b_only),
            "e3_valid": sum(r.get("status") == "ok" for r in b.values()),
            "e3_invalid": sum(r.get("status") == "invalid_response" for r in b.values()),
            "e3_errors": sum(r.get("status") == "error" for r in b.values()),
            "e1_probability_valid": parent_quality["valid"],
            "e3_probability_valid": decision_quality["valid"],
            "e1_brier": parent_quality["brier"],
            "e3_brier": decision_quality["brier"],
            "e1_ece10": parent_quality["ece10"],
            "e3_ece10": decision_quality["ece10"],
            "e1_nll": parent_quality["nll"],
            "e3_nll": decision_quality["nll"],
        })
    indexed = sorted(enumerate(rows), key=lambda v: v[1]["exact_p"])
    prev = 0
    for order, (i, item) in enumerate(indexed):
        adj = min(1.0, item["exact_p"] * (len(rows) - order))
        prev = max(prev, adj)
        rows[i]["holm_p"] = prev
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "e3_vs_e1_paired_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out_dir / "e3_vs_e1_paired_comparison.json"
    json_path.write_text(json.dumps({"correction_family": "six predefined E3 parent contrasts",
                                     "denominator": 231, "comparisons": rows},
                                    indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['parent']:17s} -> {r['specialized']:18s}: "
              f"{r['parent_correct']}/231 -> {r['specialized_correct']}/231; "
              f"delta={r['delta_accuracy_pp']:+.2f} pp; Holm p={r['holm_p']:.4g}")
    print(f"\nSaved {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
