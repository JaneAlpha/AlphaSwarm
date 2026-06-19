from __future__ import annotations

import json
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


class FactorMinerWebServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.host = host
        self.port = port
        self.project_root = Path(__file__).resolve().parents[1]
        self.run_lock = threading.Lock()
        self.run_state: dict[str, Any] = {
            "running": False,
            "started_at": None,
            "finished_at": None,
                "error": None,
                "status": None,
                "stdout": "",
                "stderr": "",
            }

    def serve(self) -> None:
        server_ref = self

        class Handler(SimpleHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send_response(302)
                    self.send_header("Location", "/checkpoints/index.html")
                    self.end_headers()
                    return
                if parsed.path == "/api/checkpoints":
                    self._send_json(server_ref.list_checkpoints())
                    return
                if parsed.path == "/api/run-state":
                    self._send_json(server_ref.run_state)
                    return
                if not self._is_allowed_static(parsed.path):
                    self.send_error(404)
                    return
                super().do_GET()

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/run-loop":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                raw_body = self.rfile.read(length).decode("utf-8") if length else ""
                body = json.loads(raw_body) if raw_body else {}
                query = parse_qs(parsed.query)
                max_iterations = int(body.get("max_iterations") or query.get("max_iterations", [3])[0])
                factors_per_round = int(body.get("factors_per_round") or query.get("factors_per_round", [3])[0])
                accepted = server_ref.start_pipeline(max_iterations, factors_per_round)
                self._send_json(accepted)

            def _send_json(self, payload: dict[str, Any] | list[dict[str, Any]], status: int = 200) -> None:
                data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _is_allowed_static(self, request_path: str) -> bool:
                relative = request_path.lstrip("/")
                if not relative:
                    return False
                candidate = (server_ref.project_root / relative).resolve()
                allowed_roots = [
                    (server_ref.project_root / "checkpoints").resolve(),
                    (server_ref.project_root / "results").resolve(),
                ]
                return candidate.is_file() and any(candidate.is_relative_to(root) for root in allowed_roots)

        handler = partial(Handler, directory=str(self.project_root))
        httpd = ThreadingHTTPServer((self.host, self.port), handler)
        print(f"MAS-FactorMiner web server: http://{self.host}:{self.port}/checkpoints/index.html", flush=True)
        httpd.serve_forever()

    def list_checkpoints(self) -> dict[str, Any]:
        archives = [{"label": "当前检查点", "prefix": ""}]
        results_root = self.project_root / "results"
        if results_root.exists():
            result_dirs = sorted(
                [item for item in results_root.iterdir() if item.is_dir()],
                key=lambda item: item.name,
                reverse=True,
            )
            archives.extend(
                {
                    "label": item.name,
                    "prefix": f"/results/{item.name}/",
                }
                for item in result_dirs
            )
        return {"archives": archives}

    def start_pipeline(self, max_iterations: int, factors_per_round: int) -> dict[str, Any]:
        with self.run_lock:
            if self.run_state["running"]:
                return {"accepted": False, "message": "pipeline is already running", "state": self.run_state}
            self.run_state = {
                "running": True,
                "started_at": None,
                "finished_at": None,
                "error": None,
                "status": None,
                "stdout": "",
                "stderr": "",
                "max_iterations": max_iterations,
                "factors_per_round": factors_per_round,
            }
            worker = threading.Thread(
                target=self._run_pipeline,
                args=(max_iterations, factors_per_round),
                daemon=True,
            )
            worker.start()
        return {"accepted": True, "message": "pipeline started", "state": self.run_state}

    def _run_pipeline(self, max_iterations: int, factors_per_round: int) -> None:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(self.project_root / "run.py"),
                    "run-loop-once",
                    "--max-iterations",
                    str(max_iterations),
                    "--factors-per-round",
                    str(factors_per_round),
                ],
                cwd=self.project_root,
                text=True,
                capture_output=True,
                timeout=None,
            )
            status = self._load_latest_status()
            if completed.returncode != 0:
                error = completed.stderr.strip() or completed.stdout.strip() or f"pipeline exited with {completed.returncode}"
                self.run_state.update(
                    {
                        "running": False,
                        "finished_at": status.get("finished_at") if status else None,
                        "error": error,
                        "status": status,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                )
                return
            self.run_state.update(
                {
                    "running": False,
                    "started_at": status.get("started_at"),
                    "finished_at": status.get("finished_at"),
                    "error": None,
                    "status": status,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
        except Exception as exc:
            self.run_state.update(
                {
                    "running": False,
                    "finished_at": None,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": None,
                    "stdout": "",
                    "stderr": "",
                }
            )

    def _load_latest_status(self) -> dict[str, Any] | None:
        status_path = self.project_root / "checkpoints" / "status.json"
        if not status_path.exists():
            return None
        return json.loads(status_path.read_text(encoding="utf-8"))
