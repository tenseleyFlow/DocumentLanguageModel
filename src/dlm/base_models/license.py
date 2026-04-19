"""License acceptance records for gated base models (Sprint 12b).

The `BaseModelSpec` schema already carries `requires_acceptance`,
`redistributable`, `license_spdx`, and `license_url` (Sprint 06). What
Sprint 12b adds is the *acceptance record* — a small Pydantic model
that stores "user X accepted license Y at time T via path Z", plus a
helper that validates an `accept_license` flag against the spec.

`LicenseAcceptance` rides on two load-bearing files:

- `manifest.json.license_acceptance` (this sprint): the per-store
  durable record; reads on every subsequent `dlm train` to verify
  the acceptance fingerprint is still present.
- Repo-level `dlm.lock.license_acceptance` (Sprint 15): the
  determinism-contract mirror; divergence between the two triggers a
  lock re-check.

The interactive prompt in `dlm init` is owned by Sprint 13; this
module ships the data types + helpers that sprint will call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from dlm.base_models.errors import GatedModelError
from dlm.base_models.schema import BaseModelSpec

AcceptanceVia = Literal["cli_flag", "interactive", "frontmatter"]


class LicenseAcceptance(BaseModel):
    """One acceptance record for a gated base.

    `via` records *how* acceptance was captured:

    - `"cli_flag"` — `--i-accept-license` on init/train (explicit).
    - `"interactive"` — `y/N` prompt (Sprint 13 UX).
    - `"frontmatter"` — persisted in `.dlm` frontmatter (Sprint 13).

    The `license_url` is captured at acceptance time so a later
    upstream URL change is auditable (the recorded URL stays the
    user's contract; warn-not-fail on drift per Sprint 12b risks).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    accepted_at: datetime
    license_url: str = Field(..., min_length=1)
    license_spdx: str = Field(..., min_length=1)
    via: AcceptanceVia


def _utcnow() -> datetime:
    """Timezone-aware UTC, microseconds dropped — matches Manifest's clock."""
    return datetime.now(UTC).replace(tzinfo=None, microsecond=0)


def is_gated(spec: BaseModelSpec) -> bool:
    """Return True iff this base requires explicit acceptance.

    Thin wrapper over `spec.requires_acceptance` that callers import
    from one named entry point, keeping the "is this model gated?"
    question symmetric with `require_acceptance` below.
    """
    return spec.requires_acceptance


def require_acceptance(
    spec: BaseModelSpec,
    *,
    accept_license: bool,
    via: AcceptanceVia,
) -> LicenseAcceptance | None:
    """Produce a `LicenseAcceptance` for `spec` or raise `GatedModelError`.

    Non-gated specs return `None` — no record needed.

    Gated specs with `accept_license=True` return a fresh acceptance
    stamped at the current UTC minute. Gated specs with
    `accept_license=False` raise `GatedModelError` so the caller (CLI)
    can surface the license URL + flag instruction.
    """
    if not is_gated(spec):
        return None
    if not accept_license:
        raise GatedModelError(spec.hf_id, spec.license_url)
    if spec.license_url is None:
        # Defensive: `requires_acceptance=True` without a URL is a
        # registry bug. Fail loud so the registry tests catch it.
        raise GatedModelError(spec.hf_id, license_url=None)
    return LicenseAcceptance(
        accepted_at=_utcnow(),
        license_url=spec.license_url,
        license_spdx=spec.license_spdx,
        via=via,
    )
