from __future__ import annotations

import json
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .pipeline import SingleRunFactorMiningPipeline


class FactorMinerDashboardServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.host = host
        self.port = port
        self.project_root = Path(__file__).resolve().parents[1]
        self.run_lock = threading.Lock()
        self.pre_run_result_names: set[str] = set()
        self.active_result_dir: Path | None = None
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
                    self.send_header("Location", "/dashboard/index.html")
                    self.end_headers()
                    return
                if parsed.path == "/api/dashboard/results":
                    self._send_json(server_ref.list_results())
                    return
                if parsed.path == "/api/dashboard/run-state":
                    server_ref.refresh_running_status()
                    self._send_json(server_ref.run_state)
                    return
                if not self._is_allowed_static(parsed.path):
                    self.send_error(404)
                    return
                super().do_GET()

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/api/dashboard/run-loop":
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
                    (server_ref.project_root / "dashboard").resolve(),
                    (server_ref.project_root / "results").resolve(),
                ]
                return candidate.is_file() and any(candidate.is_relative_to(root) for root in allowed_roots)

        handler = partial(Handler, directory=str(self.project_root))
        httpd = ThreadingHTTPServer((self.host, self.port), handler)
        print(f"MAS-FactorMiner dashboard server: http://{self.host}:{self.port}/dashboard/index.html", flush=True)
        httpd.serve_forever()

    def list_results(self) -> dict[str, Any]:
        results_root = self.project_root / "results"
        archives: list[dict[str, str]] = []
        if results_root.exists():
            result_dirs = sorted(
                [item for item in results_root.iterdir() if item.is_dir()],
                key=lambda item: item.name,
                reverse=True,
            )
            archives = [
                {
                    "label": item.name,
                    "prefix": f"/results/{item.name}/",
                }
                for item in result_dirs
            ]
        return {
            "archives": archives,
            "latest_prefix": archives[0]["prefix"] if archives else "",
        }

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
            self.pre_run_result_names = self._result_names()
            self.active_result_dir = None
            worker = threading.Thread(
                target=self._run_pipeline,
                args=(max_iterations, factors_per_round),
                daemon=True,
            )
            worker.start()
        return {"accepted": True, "message": "pipeline started", "state": self.run_state}

    def _run_pipeline(self, max_iterations: int, factors_per_round: int) -> None:
        previous_cwd = Path.cwd()
        try:
            os.chdir(self.project_root)
            try:
                SingleRunFactorMiningPipeline(
                    max_iterations=max_iterations,
                    factors_per_round=factors_per_round,
                ).run()
                status = self._load_active_or_latest_result_status()
                self.run_state.update(
                    {
                        "running": False,
                        "started_at": status.get("started_at") if status else None,
                        "finished_at": status.get("finished_at") if status else None,
                        "error": None,
                        "status": status,
                        "stdout": "",
                        "stderr": "",
                    }
                )
            finally:
                os.chdir(previous_cwd)
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

    def refresh_running_status(self) -> None:
        if not self.run_state.get("running"):
            return
        status = self._load_active_result_status()
        if not status:
            return
        self.run_state.update(
            {
                "started_at": status.get("started_at"),
                "finished_at": status.get("finished_at"),
                "status": status,
            }
        )

    def _result_names(self) -> set[str]:
        results_root = self.project_root / "results"
        if not results_root.exists():
            return set()
        return {item.name for item in results_root.iterdir() if item.is_dir()}

    def _load_active_result_status(self) -> dict[str, Any] | None:
        if self.active_result_dir is not None:
            status_path = self.active_result_dir / "status.json"
            if status_path.exists():
                return json.loads(status_path.read_text(encoding="utf-8"))
        results_root = self.project_root / "results"
        if not results_root.exists():
            return None
        new_dirs = sorted(
            [
                item
                for item in results_root.iterdir()
                if item.is_dir() and item.name not in self.pre_run_result_names
            ],
            key=lambda item: item.name,
            reverse=True,
        )
        for result_dir in new_dirs:
            status_path = result_dir / "status.json"
            if status_path.exists():
                self.active_result_dir = result_dir
                return json.loads(status_path.read_text(encoding="utf-8"))
        return None

    def _load_active_or_latest_result_status(self) -> dict[str, Any] | None:
        status = self._load_active_result_status()
        if status:
            return status
        results_root = self.project_root / "results"
        if not results_root.exists():
            return None
        result_dirs = sorted(
            [item for item in results_root.iterdir() if item.is_dir()],
            key=lambda item: item.name,
            reverse=True,
        )
        for result_dir in result_dirs:
            status_path = result_dir / "status.json"
            if status_path.exists():
                return json.loads(status_path.read_text(encoding="utf-8"))
        return None
