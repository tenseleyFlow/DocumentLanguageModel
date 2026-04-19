# Changelog

All notable changes to DocumentLanguageModel are recorded here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project targets [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Sprint 16 (this entry) — MkDocs Material documentation site,
  starter templates under `templates/`, release and docs deployment
  workflows, `CHANGELOG.md` itself.

### Pending v1.0

- Manual verification of the README quickstart on a fresh Linux +
  macOS box.
- PyPI trusted-publisher configuration + dry-run to test.pypi.org
  before tagging v1.0.

## [1.0.0] — target

First stable release. Covers Phase 0–3 of the sprint roadmap
(scaffolding through MVP release).

### Highlights

- Full v1.0 CLI: `init`, `train`, `prompt`, `export`, `pack`,
  `unpack`, `doctor`, `show`, `migrate`.
- Content-addressed store at `~/.dlm/store/<dlm_id>/` with atomic
  manifest updates and exclusive locking.
- Hardware-aware training plan (`dlm doctor`) across CUDA / MPS /
  ROCm / CPU tiers, with a refusal matrix that fails loudly on
  unsupported combinations.
- Curated base-model registry (10 entries) plus `hf:org/name`
  escape hatch with compatibility probes.
- LoRA + QLoRA training, replay-corpus retraining that retains prior
  sections, two-phase atomic version commits.
- Eval harness: val-loss, perplexity, early-stop.
- GGUF export with imatrix-calibrated quantization, explicit Go
  chat template (no fuzzy matching), embedding-row SHA verification,
  merge-safety gate against QLoRA pitfalls.
- Ollama integration: Modelfile emission, `ollama create`, smoke
  validation, closed-loop token-identity verification against the
  HF Jinja reference.
- `.dlm.pack` format: byte-identical packs, symlink / tar-bomb /
  zstd-bomb defenses, per-file SHA-256 integrity, pack-format
  migrations registry.
- Reproducibility contract: per-store `dlm.lock` with severity-table
  mismatch policy, `--strict-lock` / `--update-lock` / `--ignore-lock`
  CLI flags, determinism golden integration test.
- Documentation: getting started, `.dlm` format reference, CLI
  reference, six cookbook recipes, architecture overview,
  troubleshooting, determinism guide.
- Five starter templates: coding tutor, domain KB, writing partner,
  personal assistant, changelog.
- Weekly CI jobs: chat-template drift, slow integration suite.
- Pre-commit config: ruff, mypy `--strict`, non-slow pytest.

### Thanks

Built by following `.docs/findings.md` and the 29-sprint plan closely.
Every pitfall in the findings inventory corresponds to a test and an
explicit guardrail somewhere in the codebase.

---

The complete per-sprint history lives in `.docs/sprints/` (local to
the repo by user choice; planning artifacts stay out of git).
