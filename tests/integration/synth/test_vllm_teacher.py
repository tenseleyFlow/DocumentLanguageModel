"""Integration proof for the OpenAI-compatible `vllm-server` synth teacher."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dlm.cli.app import app
from dlm.doc.parser import parse_file
from dlm.doc.sections import SectionType

_DLM_ID = "01KPQ9X1000000000000000000"


def _write_doc(path: Path) -> None:
    path.write_text(
        "---\n"
        f"dlm_id: {_DLM_ID}\n"
        "dlm_version: 15\n"
        "base_model: smollm2-135m\n"
        "---\n"
        "DGEMM multiplies two dense matrices and can accumulate the result.\n",
        encoding="utf-8",
    )


class _CompatHandler(BaseHTTPRequestHandler):
    requests_seen: list[tuple[str, dict[str, object] | None]] = []

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        _ = format, args
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self._write_json(200, {"data": [{"id": "stub-vllm-teacher"}]})
        self.requests_seen.append((self.path, None))

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = json.loads(raw.decode("utf-8"))
        self.requests_seen.append((self.path, payload))
        self._write_json(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '[{"question":"What does DGEMM do?",'
                                '"answer":"It multiplies dense matrices."}]'
                            )
                        }
                    }
                ]
            },
        )

    def _write_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def compat_server() -> Iterator[str]:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _CompatHandler)
    except PermissionError as exc:
        pytest.skip(f"loopback bind blocked on this host: {exc}")
    _CompatHandler.requests_seen = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        address = server.server_address
        host = str(address[0])
        port = int(address[1])
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_synth_instructions_vllm_server_teacher_applies_sections(
    tmp_path: Path,
    compat_server: str,
) -> None:
    home = tmp_path / "home"
    doc = tmp_path / "doc.dlm"
    _write_doc(doc)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--home",
            str(home),
            "synth",
            "instructions",
            str(doc),
            "--teacher",
            f"vllm-server:{compat_server}",
            "--filter",
            "dedup-only",
            "--per-section",
            "1",
            "--apply",
        ],
    )

    assert result.exit_code == 0, result.output

    parsed = parse_file(doc)
    synth_sections = [
        section
        for section in parsed.sections
        if section.type is SectionType.INSTRUCTION and section.auto_synth
    ]
    assert len(synth_sections) == 1
    assert synth_sections[0].synth_teacher == f"vllm-server:{compat_server}"
    assert synth_sections[0].synth_strategy == "extraction"

    paths = [path for path, _payload in _CompatHandler.requests_seen]
    assert "/v1/models" in paths
    assert "/v1/chat/completions" in paths

    chat_payload = next(
        payload
        for path, payload in _CompatHandler.requests_seen
        if path == "/v1/chat/completions" and payload is not None
    )
    assert isinstance(chat_payload, dict)
    assert chat_payload["model"] == "stub-vllm-teacher"
    assert isinstance(chat_payload["messages"], list)
    assert chat_payload["messages"][0]["role"] == "system"
    assert chat_payload["messages"][1]["role"] == "user"
