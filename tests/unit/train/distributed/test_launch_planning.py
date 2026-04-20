"""Accelerate-launch command construction + worker-entry arg stripping."""

from __future__ import annotations

import pytest

from dlm.train.distributed.launcher import build_accelerate_cmd
from dlm.train.distributed.worker_entry import _strip_gpus_flag


class TestBuildAccelerateCmd:
    def test_basic_shape(self) -> None:
        cmd = build_accelerate_cmd(device_ids=[0, 1], cli_args=["train", "my.dlm"])
        assert cmd[0] == "accelerate"
        assert cmd[1] == "launch"
        assert "--num_processes" in cmd
        n_idx = cmd.index("--num_processes")
        assert cmd[n_idx + 1] == "2"

    def test_num_processes_matches_device_count(self) -> None:
        cmd = build_accelerate_cmd(device_ids=[0, 2, 3], cli_args=[])
        n_idx = cmd.index("--num_processes")
        assert cmd[n_idx + 1] == "3"

    def test_mixed_precision_flag_pinned(self) -> None:
        cmd = build_accelerate_cmd(device_ids=[0, 1], cli_args=[])
        mp_idx = cmd.index("--mixed_precision")
        assert cmd[mp_idx + 1] == "bf16"

    def test_mixed_precision_override(self) -> None:
        cmd = build_accelerate_cmd(device_ids=[0, 1], cli_args=[], mixed_precision="fp16")
        mp_idx = cmd.index("--mixed_precision")
        assert cmd[mp_idx + 1] == "fp16"

    def test_single_node_flags_present(self) -> None:
        cmd = build_accelerate_cmd(device_ids=[0, 1], cli_args=[])
        assert "--machine_rank" in cmd
        assert cmd[cmd.index("--machine_rank") + 1] == "0"
        assert "--num_machines" in cmd
        assert cmd[cmd.index("--num_machines") + 1] == "1"

    def test_worker_entry_is_m_target(self) -> None:
        cmd = build_accelerate_cmd(device_ids=[0, 1], cli_args=[])
        m_idx = cmd.index("-m")
        assert cmd[m_idx + 1] == "dlm.train.distributed.worker_entry"

    def test_cli_args_appended_verbatim(self) -> None:
        cmd = build_accelerate_cmd(device_ids=[0, 1], cli_args=["train", "my.dlm", "--seed", "42"])
        # The original args land at the tail, after `-m worker_entry`.
        assert cmd[-4:] == ["train", "my.dlm", "--seed", "42"]

    def test_empty_device_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            build_accelerate_cmd(device_ids=[], cli_args=[])


class TestStripGpusFlag:
    def test_separate_value_form(self) -> None:
        assert _strip_gpus_flag(["train", "--gpus", "2", "my.dlm"]) == ["train", "my.dlm"]

    def test_equals_form(self) -> None:
        assert _strip_gpus_flag(["train", "--gpus=0,1", "my.dlm"]) == ["train", "my.dlm"]

    def test_preserves_other_flags(self) -> None:
        argv = ["train", "--seed", "42", "--gpus", "all", "--max-steps", "10", "my.dlm"]
        assert _strip_gpus_flag(argv) == [
            "train",
            "--seed",
            "42",
            "--max-steps",
            "10",
            "my.dlm",
        ]

    def test_no_gpus_present_noop(self) -> None:
        argv = ["train", "my.dlm", "--seed", "7"]
        assert _strip_gpus_flag(argv) == argv
