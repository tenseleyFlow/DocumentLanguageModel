"""Shared OpenAI-compatible smoke harness."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from dlm.export.errors import TargetSmokeError
from dlm.export.smoke import smoke_openai_compat_server


def _require_loopback_bind() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except PermissionError as exc:
        pytest.skip(f"loopback bind blocked on this host: {exc}")


def _write_server_script(tmp_path: Path, *, mode: str) -> Path:
    script = tmp_path / f"fake_server_{mode}.py"
    script.write_text(
        (
            "from __future__ import annotations\n"
            "import argparse\n"
            "import json\n"
            "from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer\n"
            "\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--host', required=True)\n"
            "parser.add_argument('--port', required=True, type=int)\n"
            "parser.add_argument('--mode', required=True)\n"
            "args = parser.parse_args()\n"
            "\n"
            "if args.mode == 'exit':\n"
            "    raise SystemExit(3)\n"
            "\n"
            "class Handler(BaseHTTPRequestHandler):\n"
            "    def do_GET(self) -> None:\n"
            "        if self.path != '/v1/models':\n"
            "            self.send_response(404)\n"
            "            self.end_headers()\n"
            "            return\n"
            "        body = json.dumps({'data': [{'id': 'fake-model'}]}).encode('utf-8')\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-Type', 'application/json')\n"
            "        self.send_header('Content-Length', str(len(body)))\n"
            "        self.end_headers()\n"
            "        self.wfile.write(body)\n"
            "\n"
            "    def do_POST(self) -> None:\n"
            "        if self.path != '/v1/chat/completions':\n"
            "            self.send_response(404)\n"
            "            self.end_headers()\n"
            "            return\n"
            "        _ = self.rfile.read(int(self.headers.get('Content-Length', '0')))\n"
            "        if args.mode == 'empty':\n"
            "            payload = {'choices': [{'message': {'content': ''}}]}\n"
            "        else:\n"
            "            payload = {'choices': [{'message': {'content': 'hello from fake server'}}]}\n"
            "        body = json.dumps(payload).encode('utf-8')\n"
            "        self.send_response(200)\n"
            "        self.send_header('Content-Type', 'application/json')\n"
            "        self.send_header('Content-Length', str(len(body)))\n"
            "        self.end_headers()\n"
            "        self.wfile.write(body)\n"
            "\n"
            "    def log_message(self, format: str, *args: object) -> None:\n"
            "        return\n"
            "\n"
            "server = ThreadingHTTPServer((args.host, args.port), Handler)\n"
            "server.serve_forever()\n"
        ),
        encoding="utf-8",
    )
    return script


class TestSmokeOpenAiCompatServer:
    def test_returns_first_response_line(self, tmp_path: Path) -> None:
        _require_loopback_bind()
        script = _write_server_script(tmp_path, mode="ok")

        first_line = smoke_openai_compat_server(
            [sys.executable, str(script), "--mode", "ok", "--host", "127.0.0.1", "--port", "8000"]
        )

        assert first_line == "hello from fake server"

    def test_empty_content_raises(self, tmp_path: Path) -> None:
        _require_loopback_bind()
        script = _write_server_script(tmp_path, mode="empty")

        with pytest.raises(TargetSmokeError, match="non-empty"):
            smoke_openai_compat_server(
                [
                    sys.executable,
                    str(script),
                    "--mode",
                    "empty",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ]
            )

    def test_early_exit_raises_with_readiness_message(self, tmp_path: Path) -> None:
        _require_loopback_bind()
        script = _write_server_script(tmp_path, mode="exit")

        with pytest.raises(TargetSmokeError, match="exited before readiness"):
            smoke_openai_compat_server(
                [
                    sys.executable,
                    str(script),
                    "--mode",
                    "exit",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                startup_timeout=1.0,
            )
