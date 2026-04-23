"""Live drift checks for curated base-model registry entries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from dlm.base_models import BASE_MODELS, BaseModelSpec

_USER_AGENT = "DocumentLanguageModel/registry-refresh"
FetchText = Callable[[str], str]


@dataclass(frozen=True)
class Drift:
    """Structured diff between a local registry entry and its live sources."""

    key: str
    hf_id: str
    fields: tuple[tuple[str, str, str], ...]

    def render(self) -> str:
        lines = [f"  {self.key} ({self.hf_id})"]
        for name, pinned, observed in self.fields:
            lines.append(f"    {name:<22} {pinned!r} → {observed!r}")
        return "\n".join(lines)


def fetch_text(url: str) -> str:
    """Fetch `url` as text for provenance checks."""

    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=15) as resp:
        body = bytes(resp.read())
        charset = str(resp.headers.get_content_charset() or "utf-8")
    return body.decode(charset, errors="replace")


def check_entry(
    api: HfApi,
    entry: BaseModelSpec,
    *,
    fetch_url_text: FetchText = fetch_text,
) -> Drift | None:
    """Return a structured drift report for one curated entry, if any."""

    try:
        info = api.model_info(entry.hf_id)
    except GatedRepoError:
        return Drift(
            key=entry.key,
            hf_id=entry.hf_id,
            fields=(("gating", "readable", "now fully gated"),),
        )
    except RepositoryNotFoundError:
        return Drift(
            key=entry.key,
            hf_id=entry.hf_id,
            fields=(("repository", "present", "missing (renamed or deleted)"),),
        )

    drifted: list[tuple[str, str, str]] = []

    current_sha = info.sha
    if current_sha and current_sha != entry.revision:
        drifted.append(("revision", entry.revision, current_sha))

    if entry.refresh_check_hf_gating:
        gated = getattr(info, "gated", False)
        gated_observed = bool(gated and gated != "False")
        if gated_observed != entry.requires_acceptance:
            drifted.append(
                (
                    "requires_acceptance",
                    str(entry.requires_acceptance),
                    str(gated_observed),
                ),
            )

    if entry.provenance_url and entry.provenance_match_text:
        expected = entry.provenance_match_text
        try:
            page = fetch_url_text(entry.provenance_url)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            drifted.append(
                (
                    "provenance_url",
                    f"{entry.provenance_url} contains {expected!r}",
                    f"unreachable ({type(exc).__name__})",
                )
            )
        else:
            if expected.casefold() not in page.casefold():
                drifted.append(
                    (
                        "provenance_marker",
                        expected,
                        f"missing from {entry.provenance_url}",
                    )
                )

    return Drift(key=entry.key, hf_id=entry.hf_id, fields=tuple(drifted)) if drifted else None


def check_registry(*, fetch_url_text: FetchText = fetch_text) -> list[Drift]:
    """Check every curated entry and return drift reports."""

    api = HfApi()
    drifts: list[Drift] = []
    for entry in BASE_MODELS.values():
        drift = check_entry(api, entry, fetch_url_text=fetch_url_text)
        if drift is not None:
            drifts.append(drift)
    return drifts
