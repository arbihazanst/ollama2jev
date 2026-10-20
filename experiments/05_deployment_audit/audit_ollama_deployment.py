#!/usr/bin/env python3
"""Read-only Ollama model inventory with opt-in inference residency probes.

No pip dependencies. Python 3.10+. No installation, model pulling, deleting,
model renaming or unloading is performed. Optional probes CAN load/evict models.

Ollama sources: GET /api/tags, GET /api/ps, POST /api/show,
optional POST /api/chat or /v1/systemone. CUDA telemetry: nvidia-smi (optional).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODELS = (
    "qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b", "qwen3.5:9b", "qwen3.8:latest",
    "tev1:0.8b", "tev1:4b", "nimble-v3:9b", "clef-flash:9b", "clef:27b",
    "pplx-decider:latest",
)
PAIRINGS = (
    ("tev1:0.8b", "qwen3.5:0.8b"),
    ("tev1:4b", "qwen3.5:4b"),
    ("nimble-v3:9b", "qwen3.5:9b"),
    ("clef-flash:9b", "qwen3.5:9b"),
    ("clef:27b", "qwen3.8:latest"),
    ("pplx-decider:latest", "qwen3.8:latest"),
)
GENERAL = set(DEFAULT_MODELS[:5])
NATIVE = set(DEFAULT_MODELS[5:])


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def gib(v):
    return round(v / (1024 ** 3), 4) if isinstance(v, (int, float)) else None


def get_json(host: str, route: str, payload=None, timeout=30):
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = Request(host.rstrip("/") + route, data=data, headers=headers,
                  method="POST" if data is not None else "GET")
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as err:
        body = err.read(600).decode("utf-8", "replace")
        raise RuntimeError(f"{route}: HTTP {err.code}: {body}") from err
    except (URLError, TimeoutError) as err:
        raise RuntimeError(f"{route}: Ollama connection error: {err}") from err


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def video_gpu_snapshot():
    """NVIDIA device-level allocation, not attributable to a single model."""
    if not shutil.which("nvidia-smi"):
        return {"status": "nvidia-smi unavailable", "gpus": [], "processes": []}
    cmds = (
        ("index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu", "gpus"),
        ("gpu_uuid,pid,process_name,used_gpu_memory", "processes"),
    )
    result = {"status": "ok", "gpus": [], "processes": []}
    for query, group in cmds:
        command = ["nvidia-smi", f"--query-{ 'gpu' if group == 'gpus' else 'compute-apps'}={query}",
                   "--format=csv,noheader,nounits"]
        try:
            process = subprocess.run(command, capture_output=True, text=True,
                                     timeout=12, check=False)
            if process.returncode != 0:
                result.setdefault("warnings", []).append(
                    group + ": " + process.stderr.strip()[:250]
                )
                continue
            keys = query.split(",")
            for line in process.stdout.splitlines():
                cells = [x.strip() for x in line.split(",")]
                if len(cells) == len(keys):
                    result[group].append(dict(zip(keys, cells)))
        except (OSError, subprocess.TimeoutExpired) as err:
            result.setdefault("warnings", []).append(group + ": " + str(err))
    return result


def manifest_path(root: Path, tag: str):
    """Ollama's usual on-disk registry layout; no assumptions for custom registries."""
    name, sep, variant = tag.rpartition(":")
    if not sep:
        name, variant = tag, "latest"
    names = name.split("/")
    if len(names) == 1:
        parts = ["registry.ollama.ai", "library", names[0], variant]
    elif len(names) == 2 and "." not in names[0]:
        parts = ["registry.ollama.ai", names[0], names[1], variant]
    elif len(names) >= 3:
        parts = names + [variant]
    else:
        return None
    if any(x in ("", ".", "..") for x in parts):
        return None
    return root.joinpath("manifests", *parts)


def read_manifest_blobs(root: Path, tag: str):
    path = manifest_path(root, tag)
    if path is None or not path.is_file():
        return {"status": "manifest_not_accessible", "blobs": {}, "manifest_path": None}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        entries = [manifest.get("config") or {}] + list(manifest.get("layers") or [])
        blobs = {}
        missing = []
        for entry in entries:
            digest = entry.get("digest")
            if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", digest):
                continue
            blob = root / "blobs" / digest.replace(":", "-", 1)
            if not blob.is_file():
                missing.append(digest)
                continue
            blobs[digest] = blob.stat().st_size
        return {
            "status": "ok" if not missing else "incomplete",
            "blobs": blobs, "manifest_path": str(path),
            "referenced_blobs": len([e for e in entries if e.get("digest")]),
            "missing_blobs": missing,
        }
    except (OSError, ValueError, TypeError) as err:
        return {"status": "manifest_read_error", "blobs": {}, "error": str(err)}


def report_snapshot(host, selected, tags, show_data, root, no_gpu):
    running = get_json(host, "/api/ps").get("models", [])
    ps_by_digest = {r.get("digest"): r for r in running if r.get("digest")}
    ps_by_name = {r.get("name") or r.get("model"): r for r in running}
    records = []
    manifests = {}
    for name in selected:
        info = tags.get(name)
        if info is None:
            records.append({"model": name, "installed": False, "error": "not listed in /api/tags"})
            continue
        details = info.get("details") or {}
        show = show_data.get(name) or {}
        shown_details = show.get("details") or {}
        digest = info.get("digest")
        runtime = ps_by_digest.get(digest) or ps_by_name.get(name) or {}
        bytes_disk = info.get("size")
        bytes_runtime = runtime.get("size")
        bytes_vram = runtime.get("size_vram")
        manifest = read_manifest_blobs(root, name) if root else None
        if manifest is not None:
            manifests[name] = manifest
        records.append({
            "model": name,
            "category": "general-purpose" if name in GENERAL else "native-systemone" if name in NATIVE else "other",
            "installed": True,
            "digest": digest,
            "modified_at": info.get("modified_at"),
            "installed_reported_bytes": bytes_disk,
            "installed_reported_gib": gib(bytes_disk),
            "family": details.get("family") or shown_details.get("family"),
            "families": ",".join(details.get("families") or []),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "format": details.get("format"),
            "model_architecture": (show.get("model_info") or {}).get("general.architecture"),
            "capabilities": ",".join(show.get("capabilities") or []),
            "loaded": bool(runtime),
            "loaded_reported_bytes": bytes_runtime,
            "loaded_reported_gib": gib(bytes_runtime),
            "loaded_vram_bytes": bytes_vram,
            "loaded_vram_gib": gib(bytes_vram),
            "loaded_ram_estimate_bytes": max(0, bytes_runtime - bytes_vram)
                if isinstance(bytes_runtime, int) and isinstance(bytes_vram, int) else None,
            "context_length": runtime.get("context_length"),
            "expires_at": runtime.get("expires_at"),
            "manifest_status": manifest["status"] if manifest else "not_requested",
            "manifest_unique_blob_bytes": sum(manifest["blobs"].values())
                if manifest and manifest["status"] == "ok" else None,
            "manifest_unique_blob_gib": gib(sum(manifest["blobs"].values()))
                if manifest and manifest["status"] == "ok" else None,
        })
    return {
        "timestamp_utc": utc_now(),
        "models": records,
        "raw_running_models": running,
        "nvidia_smi": None if no_gpu else video_gpu_snapshot(),
        "manifests": manifests,
    }


def pairings(rows, manifests):
    indexed = {r["model"]: r for r in rows}
    result = []
    for native, general in PAIRINGS:
        n, g = indexed.get(native), indexed.get(general)
        if not (n and g and n.get("installed") and g.get("installed")):
            continue
        left = manifests.get(general, {})
        right = manifests.get(native, {})
        add_blobs = None
        if left.get("status") == "ok" and right.get("status") == "ok":
            add_blobs = sum(v for d, v in right["blobs"].items() if d not in left["blobs"])
        result.append({
            "general_model": general, "decision_model": native,
            "general_digest": g.get("digest"), "decision_digest": n.get("digest"),
            "general_reported_gib": g.get("installed_reported_gib"),
            "decision_reported_gib": n.get("installed_reported_gib"),
            "two_checkpoint_reported_gib": gib(
                g["installed_reported_bytes"] + n["installed_reported_bytes"]),
            "hypothetical_second_checkpoint_reported_gib": n.get("installed_reported_gib"),
            "pair_incremental_unique_blobs_gib": gib(add_blobs),
            "same_digest": g.get("digest") == n.get("digest"),
        })
    return result


def save_snapshot(out: Path, label: str, snap: dict):
    fields = [
        "model", "category", "installed", "digest", "modified_at", "installed_reported_bytes",
        "installed_reported_gib", "family", "families", "parameter_size", "quantization", "format",
        "model_architecture", "capabilities", "loaded", "loaded_reported_bytes",
        "loaded_reported_gib", "loaded_vram_bytes", "loaded_vram_gib",
        "loaded_ram_estimate_bytes", "context_length", "expires_at", "manifest_status",
        "manifest_unique_blob_bytes", "manifest_unique_blob_gib", "error",
    ]
    write_csv(out / f"{label}_models.csv", snap["models"], fields)
    snap_json = dict(snap)
    # Only report digests and sizes of blobs, not an unnecessarily large filesystem listing.
    (out / f"{label}_details.json").write_text(json.dumps(snap_json, indent=2, ensure_ascii=False), encoding="utf-8")
    smi = snap.get("nvidia_smi") or {}
    if smi.get("gpus"):
        write_csv(out / f"{label}_gpus.csv", smi["gpus"], list(smi["gpus"][0]))
    if smi.get("processes"):
        write_csv(out / f"{label}_gpu_processes.csv", smi["processes"], list(smi["processes"][0]))


def probe_chat(host, model, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one lowercase word: yes or no. Is 2 + 2 equal to 4?"}],
        "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 1},
        "logprobs": True, "top_logprobs": 20, "keep_alive": "10m",
    }
    start = time.perf_counter()
    response = get_json(host, "/api/chat", payload=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    logprobs = response.get("logprobs") or []
    top = (logprobs[0].get("top_logprobs") or []) if logprobs else []
    tokens = {x.get("token") for x in top}
    return {
        "probe": "chat", "model": model, "timestamp_utc": utc_now(),
        "wall_time_s": round(elapsed, 5),
        "load_duration_s": round(response.get("load_duration", 0) / 1e9, 5)
            if isinstance(response.get("load_duration"), (int, float)) else None,
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "served_model": response.get("model"),
        "logprobs_first_position_reported": bool(logprobs),
        "yes_no_coverage": "yes" in tokens and "no" in tokens,
    }


def probe_generation(host, model, timeout):
    """Standard generation with the same installed checkpoint used for decisions."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "In one sentence, explain what an API is."}],
        "stream": False, "think": False,
        "options": {"temperature": 0, "num_predict": 64},
        "keep_alive": "10m",
    }
    start = time.perf_counter()
    response = get_json(host, "/api/chat", payload=payload, timeout=timeout)
    return {
        "probe": "generation", "model": model, "served_model": response.get("model"),
        "timestamp_utc": utc_now(), "wall_time_s": round(time.perf_counter() - start, 5),
        "load_duration_s": round(response.get("load_duration", 0) / 1e9, 5)
            if isinstance(response.get("load_duration"), (int, float)) else None,
        "prompt_eval_count": response.get("prompt_eval_count"),
        "eval_count": response.get("eval_count"),
        "generated_content_present": bool(((response.get("message") or {}).get("content") or "").strip()),
        "generated_content_length": len(((response.get("message") or {}).get("content") or "").strip()),
        "thinking_content_present": bool((response.get("message") or {}).get("thinking")),
        "done_reason": response.get("done_reason"),
    }


def probe_native(host, model, timeout):
    payload = {
        "model": model, "state": "The customer was charged twice and requests a refund.",
        "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    }
    start = time.perf_counter()
    response = get_json(host, "/v1/systemone", payload=payload, timeout=timeout)
    elapsed = time.perf_counter() - start
    return {
        "probe": "systemone", "model": model, "timestamp_utc": utc_now(),
        "wall_time_s": round(elapsed, 5),
        "typed_answer_present": isinstance((response.get("answers") or {}).get("refund", {}).get("noul"), (float, int)),
        "usage": response.get("usage"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="http://127.0.0.1:11434", help="Ollama local server base URL")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/deployment_audit"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Exact local Ollama tags to include")
    parser.add_argument("--all-models", action="store_true", help="Inventory all installed tags")
    parser.add_argument("--models-dir", type=Path, help="Optional Ollama models storage root containing manifests/ and blobs/")
    parser.add_argument("--no-gpu", action="store_true", help="Do not call nvidia-smi")
    parser.add_argument("--probe-chat", action="append", default=[], metavar="MODEL",
                        help="OPT-IN: load and exercise general-purpose model through Ollama chat API")
    parser.add_argument("--probe-dual-use", action="append", default=[], metavar="MODEL",
                        help="OPT-IN: run standard generation AND restricted-logprob binary scoring with one existing checkpoint")
    parser.add_argument("--probe-native", action="append", default=[], metavar="MODEL",
                        help="OPT-IN: load and exercise native decision model through /v1/systemone")
    parser.add_argument("--timeout", type=float, default=180, help="Probe network timeout (seconds)")
    args = parser.parse_args()
    if args.host.startswith("https://") or not any(x in args.host for x in ("127.0.0.1", "localhost", "[::1]")):
        print("WARNING: Endpoint may be remote. Avoid unsecured remote Ollama access; local filesystem inspection is local only.", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_tags = get_json(args.host, "/api/tags").get("models", [])
    tags = {m.get("name") or m.get("model"): m for m in raw_tags}
    selected = sorted(tags) if args.all_models else list(dict.fromkeys(args.models))
    for probe in args.probe_chat + args.probe_native + args.probe_dual_use:
        if probe not in selected:
            selected.append(probe)

    root = args.models_dir or (Path(os.environ["OLLAMA_MODELS"]) if os.environ.get("OLLAMA_MODELS") else None)
    if root is not None:
        root = root.expanduser().resolve()
        if not root.is_dir():
            print(f"WARNING: local models directory is unavailable: {root}; skipping manifest inspection", file=sys.stderr)
            root = None

    show_data = {}
    errors = []
    for name in selected:
        if name not in tags:
            errors.append(f"{name}: not installed (not returned by /api/tags)")
            continue
        try:
            show_data[name] = get_json(args.host, "/api/show", {"model": name})
        except RuntimeError as err:
            errors.append(f"{name}: show endpoint unavailable: {err}")

    before = report_snapshot(args.host, selected, tags, show_data, root, args.no_gpu)
    save_snapshot(args.output_dir, "baseline", before)
    pairs = pairings(before["models"], before["manifests"])
    if pairs:
        write_csv(args.output_dir / "pairwise_checkpoint_comparison.csv", pairs,
                  list(pairs[0]))

    events = []
    for mode, names in (("chat", args.probe_chat), ("systemone", args.probe_native),
                        ("dual_use", args.probe_dual_use)):
        for index, name in enumerate(names):
            if name not in tags:
                events.append({"model": name, "probe": mode, "error": "model not installed"})
                continue
            print(f"[probe {mode}] {name}: request may load or evict model(s)...", flush=True)
            try:
                if mode == "chat":
                    result = probe_chat(args.host, name, args.timeout)
                elif mode == "systemone":
                    result = probe_native(args.host, name, args.timeout)
                else:
                    generation = probe_generation(args.host, name, args.timeout)
                    binary = probe_chat(args.host, name, args.timeout)
                    result = {
                        "probe": "dual_use", "model": name, "timestamp_utc": utc_now(),
                        "generation": generation, "binary_scoring": binary,
                        "both_paths_responded": generation["generated_content_present"] and
                            binary["logprobs_first_position_reported"] and binary["yes_no_coverage"],
                        "same_served_model_tag": generation.get("served_model") == binary.get("served_model"),
                    }
                events.append(result)
            except (RuntimeError, ValueError) as err:
                events.append({"model": name, "probe": mode, "error": str(err), "timestamp_utc": utc_now()})
            try:
                post_tags = {
                    model.get("name") or model.get("model"): model
                    for model in get_json(args.host, "/api/tags").get("models", [])
                }
                events[-1]["model_digest_before"] = (tags.get(name) or {}).get("digest")
                events[-1]["model_digest_after"] = (post_tags.get(name) or {}).get("digest")
                events[-1]["model_digest_unchanged"] = (
                    events[-1]["model_digest_before"] is not None and
                    events[-1]["model_digest_before"] == events[-1]["model_digest_after"]
                )
                events[-1]["installed_tag_set_unchanged"] = set(tags) == set(post_tags)
            except RuntimeError as err:
                events[-1]["post_probe_inventory_warning"] = str(err)
            item = events[-1]
            kind = item.get("probe")
            if "error" in item:
                probe_pass = False
            elif kind == "dual_use":
                probe_pass = bool(
                    item.get("both_paths_responded")
                    and item.get("same_served_model_tag")
                    and item.get("generation", {}).get("served_model") == name
                    and item.get("binary_scoring", {}).get("served_model") == name
                    and item.get("model_digest_unchanged")
                    and item.get("installed_tag_set_unchanged")
                )
            elif kind == "chat":
                probe_pass = bool(item.get("logprobs_first_position_reported") and
                                  item.get("yes_no_coverage") and item.get("served_model") == name)
            elif kind == "systemone":
                probe_pass = bool(item.get("typed_answer_present"))
            else:
                probe_pass = False
            item["probe_pass"] = probe_pass
            snapshot = report_snapshot(args.host, selected, tags, show_data, root, args.no_gpu)
            label = f"after_{mode}_{index+1}_{re.sub(r'[^A-Za-z0-9_-]+', '_', name)}"
            save_snapshot(args.output_dir, label, snapshot)
            events[-1]["snapshot_prefix"] = label
    try:
        server_version = get_json(args.host, "/api/version", timeout=10).get("version")
    except RuntimeError as err:
        server_version = None
        errors.append(f"Ollama version endpoint unavailable: {err}")
    report = {
        "timestamp_utc": utc_now(), "host": args.host, "ollama_version": server_version,
        "python": sys.version.split()[0],
        "platform": platform.platform(), "local_models_dir": str(root) if root else None,
        "selected": selected, "missing": [x for x in selected if x not in tags],
        "warnings": errors,
        "baseline": before,
        "pairwise_comparison": pairs,
        "probes": events,
        "notes": [
            "/api/tags size is the Ollama-reported installed model size, not guaranteed incremental physical disk use.",
            "/api/ps size and size_vram are Ollama-reported memory/residency values for currently loaded models.",
            "The difference (size - size_vram) is only an approximation, not measured RSS or exclusive system RAM.",
            "nvidia-smi memory.used measures the entire GPU, including unrelated processes and buffers; not per-model consumption.",
            "pair_incremental_unique_blobs_gib is estimated from two accessible manifests and ignores references by other installed models.",
            "Snapshots may include other active workloads and shared allocations. No controlled latency, power, fine-tuning, or deployment-time benchmark is implied.",
            "Explicit --probe-* requests may load models and evict other models; default operation is read-only.",
            "Ollama served models can share checkpoint blobs. Catalog count or sum of reported sizes is not actual freeable disk usage.",
        ],
    }
    (args.output_dir / "deployment_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "probe_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    missing = sum(not r.get("installed") for r in before["models"])
    loaded = sum(bool(r.get("loaded")) for r in before["models"])
    print(f"Saved audit to {args.output_dir.resolve()}")
    print(f"Selected models: {len(selected)} | installed: {len(selected) - missing} | loaded: {loaded} | missing: {missing}")
    print("Files: deployment_audit.json, baseline_models.csv, pairwise_checkpoint_comparison.csv (when pairs available), optional GPU CSV and probe snapshots")
    if root is None:
        print("TIP: Pass --models-dir /path/to/ollama/models for optional physical blob-size comparisons.")
    if events:
        for item in events:
            passed = bool(item.get("probe_pass"))
            print("  probe", item.get("probe"), item.get("model"),
                  "PASS" if passed else "FAIL",
                  ("error=" + item["error"]) if "error" in item else "")
    return 2 if any(not item.get("probe_pass") for item in events) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
