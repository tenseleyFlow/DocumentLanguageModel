"""Re-exports for the `dlm.cli.commands` package.

Each `*_cmd` function lives in its own submodule
(`dlm.cli.commands.<name>`); this `__init__.py` re-exports them so
`from dlm.cli.commands import <foo>_cmd` and `commands.<foo>_cmd`
keep working from `dlm.cli.app` and any test fixtures that bind
helpers by their pre-package name.

Private helpers (`_*`) are also re-exported when test fixtures
monkeypatch them through `dlm.cli.commands.<name>` — keeping the
import path stable here lets callers stay agnostic about which
submodule owns the helper.
"""

from __future__ import annotations

from dlm.cli.commands._shared import _human_size as _human_size
from dlm.cli.commands._shared import _previously_accepted as _previously_accepted
from dlm.cli.commands.cache import _parse_duration as _parse_duration
from dlm.cli.commands.cache import cache_clear_cmd as cache_clear_cmd
from dlm.cli.commands.cache import cache_prune_cmd as cache_prune_cmd
from dlm.cli.commands.cache import cache_show_cmd as cache_show_cmd
from dlm.cli.commands.doctor import doctor_cmd as doctor_cmd
from dlm.cli.commands.export import export_cmd as export_cmd
from dlm.cli.commands.harvest import harvest_cmd as harvest_cmd
from dlm.cli.commands.init import _prompt_accept_license as _prompt_accept_license
from dlm.cli.commands.init import init_cmd as init_cmd
from dlm.cli.commands.metrics import metrics_cmd as metrics_cmd
from dlm.cli.commands.metrics import metrics_watch_cmd as metrics_watch_cmd
from dlm.cli.commands.migrate import migrate_cmd as migrate_cmd
from dlm.cli.commands.pack import pack_cmd as pack_cmd
from dlm.cli.commands.preference import preference_apply_cmd as preference_apply_cmd
from dlm.cli.commands.preference import preference_list_cmd as preference_list_cmd
from dlm.cli.commands.preference import preference_mine_cmd as preference_mine_cmd
from dlm.cli.commands.preference import preference_revert_cmd as preference_revert_cmd
from dlm.cli.commands.prompt import _dispatch_audio_prompt as _dispatch_audio_prompt
from dlm.cli.commands.prompt import _dispatch_vl_prompt as _dispatch_vl_prompt
from dlm.cli.commands.prompt import prompt_cmd as prompt_cmd
from dlm.cli.commands.pull import pull_cmd as pull_cmd
from dlm.cli.commands.push import push_cmd as push_cmd
from dlm.cli.commands.repl import repl_cmd as repl_cmd
from dlm.cli.commands.serve import serve_cmd as serve_cmd
from dlm.cli.commands.show import show_cmd as show_cmd
from dlm.cli.commands.synth import synth_instructions_cmd as synth_instructions_cmd
from dlm.cli.commands.synth import synth_list_cmd as synth_list_cmd
from dlm.cli.commands.synth import synth_revert_cmd as synth_revert_cmd
from dlm.cli.commands.templates import templates_list_cmd as templates_list_cmd
from dlm.cli.commands.train import _maybe_dispatch_multi_gpu as _maybe_dispatch_multi_gpu
from dlm.cli.commands.train import _strip_gpus_from_argv as _strip_gpus_from_argv
from dlm.cli.commands.train import train_cmd as train_cmd
from dlm.cli.commands.unpack import unpack_cmd as unpack_cmd
from dlm.cli.commands.verify import verify_cmd as verify_cmd


def _stub(sprint: str, subject: str) -> None:
    """Raise a clear unimplemented error for any v1 subcommand still pending.

    Kept around because a unit test asserts the error message shape, and
    because future v2/v3 subcommand wiring may want a stable stub
    helper to register a not-yet-implemented surface in `--help`.
    """
    raise NotImplementedError(
        f"`{subject}` is not implemented yet (owned by Sprint {sprint}).",
    )
