"""Coverage-oriented command tests for the remaining CLI surface."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.exceptions import Exit as ClickExit
from typer.testing import CliRunner

from dlm.cli import commands
from dlm.cli.app import app
from dlm.share.signing import VerifyStatus
from dlm.store.paths import for_dlm


def _write_minimal_dlm(
    path: Path,
    *,
    dlm_id: str = "01KPQ9M3" + "0" * 18,
    base_model: str = "smollm2-135m",
) -> None:
    path.write_text(
        f"---\ndlm_id: {dlm_id}\ndlm_version: 12\nbase_model: {base_model}\n---\nbody\n",
        encoding="utf-8",
    )


def _joined_output(result: object) -> str:
    text = getattr(result, "output", "") + getattr(result, "stderr", "")
    return " ".join(text.split())


class TestMetricsAndDoctor:
    def test_metrics_run_id_json_and_csv(self, tmp_path: Path, monkeypatch: Any) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)
        monkeypatch.setenv("DLM_HOME", str(tmp_path / "home"))

        run = SimpleNamespace(
            run_id=7,
            phase="sft",
            seed=42,
            status="ok",
            started_at="2026-04-21T10:00:00Z",
            ended_at="2026-04-21T10:01:00Z",
        )
        steps = [SimpleNamespace(step=1, loss=0.5, lr=1e-4, grad_norm=0.9)]
        evals = [SimpleNamespace(step=1, val_loss=0.4, perplexity=1.5)]

        monkeypatch.setattr("dlm.metrics.queries.recent_runs", lambda *args, **kwargs: [run])
        monkeypatch.setattr("dlm.metrics.queries.steps_for_run", lambda *args, **kwargs: steps)
        monkeypatch.setattr("dlm.metrics.queries.evals_for_run", lambda *args, **kwargs: evals)
        monkeypatch.setattr(
            "dlm.metrics.queries.runs_to_dict",
            lambda runs: [
                {
                    "run_id": r.run_id,
                    "phase": r.phase,
                    "seed": r.seed,
                    "status": r.status,
                }
                for r in runs
            ],
        )
        monkeypatch.setattr(
            "dlm.metrics.queries.steps_to_dict",
            lambda rows: [{"step": r.step, "loss": r.loss} for r in rows],
        )
        monkeypatch.setattr(
            "dlm.metrics.queries.evals_to_dict",
            lambda rows: [{"step": r.step, "val_loss": r.val_loss} for r in rows],
        )

        import sys
        from io import StringIO

        old_stdout = sys.stdout
        try:
            json_buf = StringIO()
            sys.stdout = json_buf
            commands.metrics_cmd(doc, json_out=True, run_id=7)
        finally:
            sys.stdout = old_stdout
        payload = json.loads(json_buf.getvalue())
        assert payload["run"]["run_id"] == 7
        assert payload["steps"][0]["step"] == 1
        assert payload["evals"][0]["val_loss"] == 0.4

        old_stdout = sys.stdout
        try:
            csv_buf = StringIO()
            sys.stdout = csv_buf
            commands.metrics_cmd(doc, csv_out=True, run_id=7)
        finally:
            sys.stdout = old_stdout
        csv_text = csv_buf.getvalue()
        assert "step,loss,lr,grad_norm,val_loss" in csv_text
        assert "1,0.5,0.0001,0.9,0.4" in csv_text

    def test_metrics_since_parse_and_watch(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        doc = tmp_path / "doc.dlm"
        home = tmp_path / "home"
        _write_minimal_dlm(doc)
        monkeypatch.setenv("DLM_HOME", str(home))

        with pytest.raises(ClickExit) as excinfo:
            commands.metrics_cmd(doc, since="bogus")
        assert excinfo.type is ClickExit
        err = capsys.readouterr().err
        assert "not an integer+unit" in err

        monkeypatch.setattr("dlm.metrics.queries.latest_run_id", lambda *_: 7)
        monkeypatch.setattr(
            "dlm.metrics.queries.steps_for_run",
            lambda *args, **kwargs: [SimpleNamespace(step=3, loss=0.25, lr=1e-4, grad_norm=0.8)],
        )
        monkeypatch.setattr(
            "dlm.metrics.queries.evals_for_run",
            lambda *args, **kwargs: [SimpleNamespace(step=3, val_loss=0.2, perplexity=1.2)],
        )

        def _interrupt(*_: object) -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr("time.sleep", _interrupt)
        commands.metrics_watch_cmd(doc, poll_seconds=0.0)
        out = capsys.readouterr().out
        assert "following run_id=7" in out
        assert "eval @ step 3" in out
        assert "bye" in out

    def test_doctor_json_and_text(self, monkeypatch: Any, capsys: Any) -> None:
        fake = SimpleNamespace(to_dict=lambda: {"plan": "ok"})
        monkeypatch.setattr("dlm.hardware.doctor", lambda: fake)
        monkeypatch.setattr("dlm.hardware.render_text", lambda result: f"doctor text: {result!r}")

        commands.doctor_cmd(json_out=True)
        json_out = capsys.readouterr().out
        assert '"plan": "ok"' in json_out

        commands.doctor_cmd(json_out=False)
        text_out = capsys.readouterr().out
        assert "doctor text" in text_out


class TestPackUnpackVerify:
    def test_pack_and_unpack_success(self, tmp_path: Path, monkeypatch: Any) -> None:
        runner = CliRunner()
        pack_out = tmp_path / "bundle.dlm.pack"
        monkeypatch.setattr(
            "dlm.pack.packer.pack",
            lambda *args, **kwargs: SimpleNamespace(
                path=pack_out,
                bytes_written=5 * 1024 * 1024,
                content_type="application/dlm-pack",
            ),
        )
        pack_result = runner.invoke(
            app, ["pack", str(tmp_path / "doc.dlm"), "--out", str(pack_out)]
        )
        assert pack_result.exit_code == 0, pack_result.output
        assert "packed:" in pack_result.output
        assert "application/dlm-pack" in pack_result.output

        monkeypatch.setattr(
            "dlm.pack.unpacker.unpack",
            lambda *args, **kwargs: SimpleNamespace(
                dlm_path=tmp_path / "restored.dlm",
                store_path=tmp_path / "home" / "store" / "01XYZ",
                dlm_id="01XYZ",
                header=SimpleNamespace(pack_format_version=2),
                applied_migrations=[2, 3],
            ),
        )
        unpack_result = runner.invoke(
            app,
            ["unpack", str(pack_out), "--out", str(tmp_path / "restore")],
        )
        assert unpack_result.exit_code == 0, unpack_result.output
        assert "unpacked:" in unpack_result.output
        assert "migrated: v2" in unpack_result.output

    def test_verify_success_and_unsigned(self, tmp_path: Path, monkeypatch: Any) -> None:
        runner = CliRunner()
        pack_path = tmp_path / "bundle.dlm.pack"
        pack_path.write_bytes(b"pack")

        provenance = SimpleNamespace(
            adapter_sha256="a" * 64,
            base_revision="main",
            corpus_root_sha256="b" * 64,
            signed_at="2026-04-21T12:00:00Z",
        )
        monkeypatch.setattr(
            "dlm.pack.unpacker.read_pack_member_bytes", lambda *args, **kwargs: b"{}"
        )
        monkeypatch.setattr("dlm.share.provenance.load_provenance_json", lambda *_: provenance)
        monkeypatch.setattr(
            "dlm.share.provenance.verify_provenance",
            lambda *args, **kwargs: SimpleNamespace(
                signer_fingerprint="FINGERPRINT",
                trusted_key_path=tmp_path / "trusted.pub",
                tofu_recorded=True,
            ),
        )

        result = runner.invoke(app, ["verify", str(pack_path), "--trust-on-first-use"])
        assert result.exit_code == 0, result.output
        assert "verified:" in result.output
        assert "recorded new trust entry" in result.output

        monkeypatch.setattr(
            "dlm.pack.unpacker.read_pack_member_bytes",
            lambda *args, **kwargs: None,
        )
        unsigned = runner.invoke(app, ["verify", str(pack_path)])
        assert unsigned.exit_code == 1, unsigned.output
        assert "is unsigned" in unsigned.output


class TestMigrateTemplatesShareAndServe:
    def test_migrate_templates_push_pull_and_serve(self, tmp_path: Path, monkeypatch: Any) -> None:
        runner = CliRunner()
        doc = tmp_path / "doc.dlm"
        home = tmp_path / "home"
        _write_minimal_dlm(doc)

        monkeypatch.setattr(
            "dlm.doc.migrate.migrate_file",
            lambda *args, **kwargs: SimpleNamespace(
                applied=[11],
                target_version=12,
                backup_path=tmp_path / "doc.dlm.bak",
            ),
        )
        migrate_result = runner.invoke(app, ["migrate", str(doc)])
        assert migrate_result.exit_code == 0, migrate_result.output
        assert "migrated:" in migrate_result.output

        template = SimpleNamespace(
            name="starter",
            meta=SimpleNamespace(
                title="Starter",
                domain_tags=("docs",),
                recommended_base="smollm2-135m",
                expected_steps=25,
                expected_duration={"minutes": 5},
                summary="Starter template",
                sample_prompts=("hello",),
            ),
        )
        monkeypatch.setattr("dlm.templates.list_bundled", lambda: [template])
        templates_result = runner.invoke(app, ["templates", "list", "--json"])
        assert templates_result.exit_code == 0, templates_result.output
        assert '"name":"starter"' in templates_result.output.replace(" ", "")

        templates_text = runner.invoke(app, ["templates", "list"])
        assert templates_text.exit_code == 0, templates_text.output
        assert "Starter" in templates_text.output
        assert "smollm2-135m" in templates_text.output

        from dlm.share.sinks import SinkKind

        monkeypatch.setattr(
            "dlm.share.push",
            lambda *args, **kwargs: SimpleNamespace(
                destination="hf:org/repo",
                bytes_sent=3 * 1024 * 1024,
                sink_kind=SinkKind.HF,
                detail="done",
            ),
        )
        push_result = runner.invoke(
            app,
            ["push", str(doc), "--to", "hf:org/repo"],
        )
        assert push_result.exit_code == 0, push_result.output
        assert "pushed:" in push_result.output
        assert "dlm pull hf:org/repo" in push_result.output

        monkeypatch.setattr(
            "dlm.share.pull",
            lambda *args, **kwargs: SimpleNamespace(
                source="hf:org/repo",
                dlm_path=tmp_path / "restored.dlm",
                bytes_received=2 * 1024 * 1024,
                verification=SimpleNamespace(
                    status=VerifyStatus.VERIFIED,
                    key_path=tmp_path / "trusted.pub",
                    detail="",
                ),
            ),
        )
        pull_result = runner.invoke(app, ["pull", "hf:org/repo"])
        assert pull_result.exit_code == 0, pull_result.output
        assert "pulled:" in pull_result.output
        assert "verified:" in pull_result.output

        store = for_dlm("01KPQ9M3" + "0" * 18, home=home)
        store.ensure_layout()
        store.manifest.write_text("{}", encoding="utf-8")

        def _fake_pack(path: Path, *, out: Path) -> None:
            out.write_bytes(b"pack")

        handle = SimpleNamespace(
            bind_host="127.0.0.1",
            port=7337,
            peer_url="peer://127.0.0.1:7337/01KPQ9M3",
            wait_shutdown=lambda: None,
        )
        monkeypatch.setattr("dlm.pack.packer.pack", _fake_pack)
        monkeypatch.setattr("dlm.share.serve", lambda *args, **kwargs: handle)
        serve_result = runner.invoke(
            app,
            ["--home", str(home), "serve", str(doc)],
        )
        assert serve_result.exit_code == 0, serve_result.output
        assert "serving:" in serve_result.output
        assert "peer URL:" in serve_result.output

    def test_templates_refresh_empty_gallery_and_pull_statuses(
        self,
        tmp_path: Path,
        monkeypatch: Any,
    ) -> None:
        runner = CliRunner()

        from dlm.templates.fetcher import RemoteFetchUnavailable

        monkeypatch.setattr(
            "dlm.templates.fetcher.fetch_all",
            lambda *args, **kwargs: (_ for _ in ()).throw(RemoteFetchUnavailable("offline")),
        )
        monkeypatch.setattr("dlm.templates.fetcher.cache_dir", lambda: tmp_path / "cache")
        monkeypatch.setattr("dlm.templates.list_bundled", lambda: [])
        empty = runner.invoke(app, ["templates", "list", "--refresh"])
        assert empty.exit_code == 1, empty.output
        joined = _joined_output(empty)
        assert "Falling back to the bundled gallery" in joined
        assert "no bundled templates found" in joined

        monkeypatch.setattr(
            "dlm.share.pull",
            lambda *args, **kwargs: SimpleNamespace(
                source="hf:org/repo",
                dlm_path=tmp_path / "restored.dlm",
                bytes_received=1 * 1024 * 1024,
                verification=SimpleNamespace(
                    status=VerifyStatus.UNVERIFIED,
                    key_path=None,
                    detail="missing key",
                ),
            ),
        )
        unverified = runner.invoke(app, ["pull", "hf:org/repo"])
        assert unverified.exit_code == 0, unverified.output
        assert "unverified:" in unverified.output

        monkeypatch.setattr(
            "dlm.share.pull",
            lambda *args, **kwargs: SimpleNamespace(
                source="hf:org/repo",
                dlm_path=tmp_path / "restored.dlm",
                bytes_received=1 * 1024 * 1024,
                verification=SimpleNamespace(
                    status=VerifyStatus.UNSIGNED,
                    key_path=None,
                    detail="",
                ),
            ),
        )
        unsigned = runner.invoke(app, ["pull", "hf:org/repo"])
        assert unsigned.exit_code == 0, unsigned.output
        assert "unsigned" in unsigned.output


class TestExportAndCacheCoverage:
    def test_export_standard_path_success_and_errors(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        doc = tmp_path / "doc.dlm"
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--home",
                str(tmp_path / "home"),
                "init",
                str(doc),
                "--base",
                "smollm2-135m",
            ],
        )
        assert result.exit_code == 0, result.output

        cached = SimpleNamespace(path=tmp_path / "base")
        monkeypatch.setattr("dlm.base_models.download_spec", lambda *args, **kwargs: cached)
        monkeypatch.setattr(
            "dlm.export.run_export",
            lambda *args, **kwargs: SimpleNamespace(
                cached=True,
                export_dir=tmp_path / "exports" / "Q4_K_M",
                artifacts=[SimpleNamespace(name="base.gguf"), SimpleNamespace(name="adapter.gguf")],
                ollama_name="my-model",
                ollama_version=1,
                smoke_output_first_line="hello",
            ),
        )
        export_ok = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "export", str(doc), "--skip-ollama"],
        )
        assert export_ok.exit_code == 0, export_ok.output
        assert "exported:" in export_ok.output
        assert "ollama:" in export_ok.output
        assert "smoke:" in export_ok.output

        monkeypatch.setattr(
            "dlm.base_models.download_spec",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline cache miss")),
        )
        cache_miss = runner.invoke(app, ["--home", str(tmp_path / "home"), "export", str(doc)])
        assert cache_miss.exit_code == 1, cache_miss.output
        assert "base model not in local cache" in cache_miss.output

        from dlm.export.ollama.errors import OllamaBinaryNotFoundError

        monkeypatch.setattr("dlm.base_models.download_spec", lambda *args, **kwargs: cached)
        monkeypatch.setattr(
            "dlm.export.run_export",
            lambda *args, **kwargs: (_ for _ in ()).throw(OllamaBinaryNotFoundError("missing")),
        )
        no_ollama = runner.invoke(app, ["--home", str(tmp_path / "home"), "export", str(doc)])
        assert no_ollama.exit_code == 1, no_ollama.output
        assert "install from https://ollama.com/download" in no_ollama.output

    def test_cache_show_prune_and_clear(self, tmp_path: Path, monkeypatch: Any) -> None:
        doc = tmp_path / "doc.dlm"
        _write_minimal_dlm(doc)

        class _FakeCache:
            def __init__(self, entry_count: int, total_bytes: int) -> None:
                self.entry_count = entry_count
                self.total_bytes = total_bytes

            def prune(self, *, older_than_seconds: float) -> int:
                self.pruned_seconds = older_than_seconds
                return 2

            def clear(self) -> int:
                return 5

            def save_manifest(self) -> None:
                return None

        fake_cache = _FakeCache(entry_count=5, total_bytes=2048)
        monkeypatch.setattr("dlm.directives.cache.TokenizedCache.open", lambda *_: fake_cache)
        monkeypatch.setattr(
            "dlm.metrics.queries.latest_tokenization",
            lambda *_: SimpleNamespace(run_id=9, hit_rate=0.75, cache_hits=3, cache_misses=1),
        )

        runner = CliRunner()
        show_json = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "cache", "show", str(doc), "--json"],
        )
        assert show_json.exit_code == 0, show_json.output
        assert '"entry_count": 5' in show_json.output

        monkeypatch.setattr("dlm.metrics.queries.latest_tokenization", lambda *_: None)
        show_text = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "cache", "show", str(doc)],
        )
        assert show_text.exit_code == 0, show_text.output
        assert "Cache for" in show_text.output
        assert "no tokenization runs yet" in show_text.output

        prune_bad = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "cache", "prune", str(doc), "--older-than", "bad"],
        )
        assert prune_bad.exit_code == 2, prune_bad.output
        assert "invalid --older-than" in prune_bad.output

        prune_ok = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "cache", "prune", str(doc)],
        )
        assert prune_ok.exit_code == 0, prune_ok.output
        assert "older than 90d" in prune_ok.output

        monkeypatch.setattr("typer.confirm", lambda prompt: False)
        clear_cancel = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "cache", "clear", str(doc)],
        )
        assert clear_cancel.exit_code == 0, clear_cancel.output
        assert "clear cancelled" in clear_cancel.output

        clear_force = runner.invoke(
            app,
            ["--home", str(tmp_path / "home"), "cache", "clear", str(doc), "--force"],
        )
        assert clear_force.exit_code == 0, clear_force.output
        assert "cleared 5" in clear_force.output

    def test_parse_duration_helper(self) -> None:
        assert commands._parse_duration("30m") == 1800.0
        assert commands._parse_duration("2h") == 7200.0
        assert commands._parse_duration("1d") == 86400.0
        assert commands._parse_duration("7x") is None
