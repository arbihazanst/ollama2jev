import json
import time
from pathlib import Path

from llm2jev import (
    Choice,
    JevRequest,
    LLM2Jev,
    Noul,
    OllamaBackend,
    Score,
)

MODELS = [
    "qwen3.5:0.8b",
    "qwen3.5:9b",
    "llama3.2:1b",
    "llama3.2:3b",
    "granite4.2:3b",
    "granite4.2:8b",
    "gemma4:e2b",
    "gemma4:e4b"
]

DATASETS = [
    "../jevbench/datasets/public/original.jsonl",
    "../jevbench/datasets/public/easy.jsonl",
    "../jevbench/datasets/public/hard.jsonl",
]


def load_balanced_tasks(paths, per_type=5):
    selected = {
        "choice": [],
        "noul": [],
        "score": [],
    }

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                task = json.loads(line)
                qtype = task["question"]["type"]

                if (
                    qtype in selected
                    and len(selected[qtype]) < per_type
                ):
                    selected[qtype].append(task)

                if all(
                    len(items) >= per_type
                    for items in selected.values()
                ):
                    break

        if all(
            len(items) >= per_type
            for items in selected.values()
        ):
            break

    tasks = (
        selected["choice"]
        + selected["noul"]
        + selected["score"]
    )

    return tasks


def make_question(question_spec):
    qtype = question_spec["type"]

    instructions = question_spec.get("instructions")
    criteria = question_spec.get("criteria")

    if qtype == "choice":
        return Choice(
            instructions=instructions,
            criteria=criteria,
        )

    if qtype == "score":
        return Score(
            instructions=instructions,
            criteria=criteria,
        )

    if qtype == "noul":
        return Noul(
            instructions=instructions,
            criteria=criteria,
        )

    raise ValueError(
        f"Unsupported question type: {qtype}"
    )


def extract_prediction(answer, question_type):
    if question_type == "choice":
        return answer.choice

    if question_type == "noul":
        return "yes" if answer.noul >= 0.5 else "no"

    if question_type == "score":
        # JevBench accuracy uses argmax over score levels.
        return max(
            answer.probabilities,
            key=answer.probabilities.get,
        )

    raise ValueError(
        f"Unsupported question type: {question_type}"
    )


def normalize_expected(expected, question_type):
    if question_type == "score":
        return int(expected)

    return expected


def run_task(task, model, backend):
    question_spec = task["question"]
    question_type = question_spec["type"]

    question = make_question(question_spec)

    request = JevRequest(
        model=model,
        state=task["state"],
        questions={
            "decision": question
        },
    )

    start = time.perf_counter()

    response = LLM2Jev(
        backend=backend
    ).evaluate(request)

    latency_s = time.perf_counter() - start

    answer = response.answers["decision"]

    prediction = extract_prediction(
        answer,
        question_type,
    )

    expected = normalize_expected(
        task["expected"],
        question_type,
    )

    correct = prediction == expected

    return {
        "id": task["id"],
        "family": task.get("family"),
        "question_type": question_type,

        "expected": expected,
        "prediction": prediction,
        "correct": correct,

        "latency_s": latency_s,

        "input_tokens":
            response.usage.input_tokens,

        "output_tokens":
            response.usage.output_tokens,

        "diagnostics":
            backend.get_diagnostics(),

        "diagnostic_summary":
            backend.diagnostic_summary(),
    }

def run_model(model, tasks):
    print()
    print("#" * 80)
    print(f"MODEL: {model}")
    print("#" * 80)

    backend = OllamaBackend(
        top_logprobs=20,
        think=False,
        keep_alive="10m",
    )

    results = []

    for i, task in enumerate(tasks, start=1):
        print()
        print("=" * 70)
        print(f"[{i}/{len(tasks)}] {task['id']}")
        print("=" * 70)

        try:
            result = run_task(
                task,
                model,
                backend,
            )

            results.append(result)

            status = (
                "CORRECT"
                if result["correct"]
                else "WRONG"
            )

            print("Type:", result["question_type"])
            print("Expected:", result["expected"])
            print("Prediction:", result["prediction"])
            print("Result:", status)
            print(
                "Latency:",
                f"{result['latency_s']:.3f}s"
            )
            print(
                "Coverage:",
                f"{result['diagnostic_summary']['coverage_rate'] * 100:.1f}%"
            )

        except Exception as error:
            print(
                "FAILED:",
                type(error).__name__,
                error,
            )

            results.append({
                "id": task["id"],
                "question_type":
                    task["question"]["type"],
                "expected": task["expected"],
                "error": str(error),
                "correct": False,
                "diagnostics":
                    backend.get_diagnostics(),
            })

    successful = [
        r for r in results
        if "prediction" in r
    ]

    correct = sum(
        r["correct"]
        for r in successful
    )

    total = len(results)

    print()
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print("Model:", model)
    print("Tasks:", total)
    print("Successful:", len(successful))
    print("Correct:", correct)

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    type_summary = {}

    for qtype in ["choice", "noul", "score"]:
        rows = [
            r for r in results
            if r.get("question_type") == qtype
        ]

        if rows:
            correct_type = sum(
                r.get("correct", False)
                for r in rows
            )

            type_summary[qtype] = {
                "tasks": len(rows),
                "correct": correct_type,
                "accuracy": correct_type / len(rows),
            }

    safe_model_name = (
        model
        .replace(":", "-")
        .replace("/", "-")
    )

    output_path = Path(
        f"ollama2jev_jevbench_{safe_model_name}_full.jsonl"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in results:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "Saved results to:",
        output_path
    )

    return {
        "model": model,
        "tasks": total,
        "successful": len(successful),
        "correct": correct,
        "accuracy": accuracy,
        "by_type": type_summary,
    }

def main():

    #print(
    #    f"Loading balanced JevBench sample: "
    #    f"{PER_TYPE} choice + "
    #    f"{PER_TYPE} noul + "
    #    f"{PER_TYPE} score"
    #)

    #tasks = load_balanced_tasks(
    #    DATASETS,
    #    per_type=PER_TYPE,
    #)

    #print(
    #    f"Total selected tasks: {len(tasks)}"
    #)

    print("Loading full public JevBench dataset...")

    tasks = []

    for path in DATASETS:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                tasks.append(json.loads(line))
    
    type_counts = {}

    for task in tasks:
        qtype = task["question"]["type"]
        type_counts[qtype] = (
            type_counts.get(qtype, 0) + 1
        )

    print("Type counts:", type_counts)
    
    print(
        f"Total selected tasks: {len(tasks)}"
    )
    
    summaries = []

    for model in MODELS:
        summary = run_model(
            model,
            tasks,
        )

        summaries.append(summary)

    print()
    print("#" * 80)
    print("FINAL SUMMARY")
    print("#" * 80)

    for summary in summaries:
        print()
        print(summary["model"])

        print(
            f"  overall: "
            f"{summary['accuracy']:.3f} "
            f"({summary['correct']}/{summary['tasks']})"
        )

        for qtype, stats in summary["by_type"].items():
            print(
                f"  {qtype:6s}: "
                f"{stats['accuracy']:.3f} "
                f"({stats['correct']}/{stats['tasks']})"
            )

    summary_path = Path(
        "ollama2jev_jevbench_model_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summaries,
            f,
            indent=2,
        )

    print()
    print(
        "Saved overall summary to:",
        summary_path
    )


if __name__ == "__main__":
    main()
