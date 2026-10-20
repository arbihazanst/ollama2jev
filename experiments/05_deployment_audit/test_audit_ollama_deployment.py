#!/usr/bin/env python3
"""Local mock tests; never connects to a real Ollama server."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).with_name("audit_ollama_deployment.py")
spec = importlib.util.spec_from_file_location("deployment", SCRIPT)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, response):
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/api/tags":
            self.send({"models": [
                {"name":"qwen3.5:9b", "digest": HASH_A, "size": 1234,
                 "details": {"parameter_size": "9B", "quantization_level": "Q4", "family": "qwen35"}},
                {"name":"nimble-v3:9b", "digest": HASH_B, "size": 4567,
                 "details": {"parameter_size": "9B", "quantization_level": "Q4", "family": "qwen35"}},
            ]})
        elif self.path == "/api/version":
            self.send({"version":"0.40.5"})
        elif self.path == "/api/ps":
            self.send({"models": [{"name": "qwen3.5:9b", "digest": HASH_A,
                                    "size": 1700, "size_vram": 1200, "context_length": 8192}]})
        else:
            self.send({"error":"unknown"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n).decode("utf-8"))
        if self.path == "/api/show":
            self.send({"details": {"family": "qwen35"},
                       "model_info": {"general.architecture": "qwen35"}, "capabilities": ["completion"]})
        elif self.path == "/api/chat":
            res = {"model": data["model"], "message":{"content":"yes"},
                   "eval_count":1, "load_duration":0, "prompt_eval_count":10}
            if data.get("logprobs"):
                res["logprobs"] = [{"top_logprobs": [
                    {"token":"yes","logprob":-0.2}, {"token":"no","logprob":-1.2}]}]
            self.send(res)
        elif self.path == "/v1/systemone":
            self.send({"model":data["model"], "answers":{"refund":{"type":"noul", "noul":0.7}},
                       "usage":{"input_tokens":15,"output_tokens":0}})
        else:
            self.send({"error":"unknown"})

class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_inventory_and_loaded_memory(self):
        tags = {x["name"]:x for x in app.get_json(self.host,"/api/tags")["models"]}
        with patch.object(app, "video_gpu_snapshot", return_value={"gpus":[],"processes":[]}):
            s = app.report_snapshot(self.host, list(tags), tags, {k:app.get_json(self.host,"/api/show",{"model":k}) for k in tags}, None, False)
        q=s["models"][0]
        self.assertEqual(q["digest"], HASH_A)
        self.assertEqual(q["installed_reported_bytes"], 1234)
        self.assertEqual(q["loaded_vram_bytes"], 1200)
        self.assertEqual(q["loaded_ram_estimate_bytes"], 500)
        self.assertTrue(q["loaded"])
        self.assertFalse(s["models"][1]["loaded"])

    def test_scoring_generation_systemone(self):
        b=app.probe_chat(self.host,"qwen3.5:9b",10)
        self.assertTrue(b["yes_no_coverage"])
        self.assertTrue(app.probe_generation(self.host,"qwen3.5:9b",10)["generated_content_present"])
        self.assertTrue(app.probe_native(self.host,"nimble-v3:9b",10)["typed_answer_present"])

    def test_end_to_end_cli_default_and_dual_use(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "results"
            cmd = [sys.executable, str(SCRIPT), "--host", self.host,
                   "--output-dir", str(out), "--no-gpu",
                   "--models", "qwen3.5:9b", "nimble-v3:9b"]
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
            self.assertEqual(p.returncode, 0, p.stderr)
            audit = json.loads((out/"deployment_audit.json").read_text())
            self.assertEqual(audit["ollama_version"], "0.40.5")
            self.assertEqual(audit["probes"], [])
            self.assertTrue((out/"baseline_models.csv").is_file())
            self.assertTrue((out/"pairwise_checkpoint_comparison.csv").is_file())
            cmd += ["--probe-dual-use", "qwen3.5:9b"]
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
            self.assertEqual(p.returncode, 0, p.stderr)
            audit = json.loads((out/"deployment_audit.json").read_text())
            self.assertTrue(audit["probes"][0]["both_paths_responded"])
            self.assertTrue(audit["probes"][0]["same_served_model_tag"])
            self.assertTrue(audit["probes"][0]["model_digest_unchanged"])
            self.assertTrue(audit["probes"][0]["installed_tag_set_unchanged"])
            self.assertTrue((out/"after_dual_use_1_qwen3_5_9b_models.csv").is_file())

    def test_manifests_and_deduplication(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/"blobs").mkdir()
            for x,num in [(HASH_A,100),(HASH_B,40),(HASH_C,80)]:
                (root/"blobs"/f"sha256-{x}").write_bytes(b"x"*num)
            for name, hashes in [("qwen3.5:9b", [HASH_A,HASH_B]),("nimble-v3:9b", [HASH_A,HASH_C])]:
                f=app.manifest_path(root,name)
                f.parent.mkdir(parents=True,exist_ok=True)
                f.write_text(json.dumps({"config":{"digest":"sha256:"+hashes[0]},
                        "layers":[{"digest":"sha256:"+hashes[1]}]}))
            a=app.read_manifest_blobs(root,"qwen3.5:9b")
            b=app.read_manifest_blobs(root,"nimble-v3:9b")
            self.assertEqual(a["status"],"ok")
            fake=[{"model":"qwen3.5:9b","installed":True,"digest":HASH_A,
                   "installed_reported_bytes":140,"installed_reported_gib":app.gib(140)},
                  {"model":"nimble-v3:9b","installed":True,"digest":HASH_B,
                   "installed_reported_bytes":180,"installed_reported_gib":app.gib(180)}]
            pairs=app.pairings(fake,{"qwen3.5:9b":a,"nimble-v3:9b":b})
            self.assertEqual(len(pairs),1)
            self.assertEqual(pairs[0]["pair_incremental_unique_blobs_gib"],app.gib(80))

if __name__=="__main__":
    unittest.main(verbosity=2)
