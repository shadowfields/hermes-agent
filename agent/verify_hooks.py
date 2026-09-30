"""Verification-loop helpers for the ``pre_verify`` round-end gate.

After code edits the loop fires ``pre_verify`` (directives resolved by
:func:`hermes_cli.plugins.get_pre_verify_continue_message`). The shipped coding
guidance rides on the evidence-based verification-stop nudge rather than a second
default stop gate, so default token cost stays tied to the "missing verification
evidence" decision while ``pre_verify`` remains free for user/plugin policy.
"""

from __future__ import annotations

from typing import Any, Optional

from utils import is_truthy_value

DEFAULT_MAX_VERIFY_NUDGES = 3

# Appended to the verification-stop nudge when code lacks fresh evidence. Mirrors
# the user-facing "clean your work" workflow without adding its own model turn.
CODING_VERIFY_GUIDANCE = (
    "[Coding] Before verification or calling the work done: simplify the changed "
    "code, remove duplicated logic and dead or experimental branches, match the "
    "surrounding code style, then run the smallest relevant checks. For creative "
    "UI or visual work, defer tests and linters until the user accepts the design "
    "or you are about to commit. Before every commit, inspect the diff for scope "
    "creep and repeat the relevant checks."
)


def max_verify_nudges(config: Optional[dict[str, Any]] = None) -> int:
    """Bound on consecutive ``pre_verify`` continue directives per turn (>= 0)."""
    try:
        return max(0, int(_agent_cfg(config).get("max_verify_nudges")))
    except (TypeError, ValueError):
        return DEFAULT_MAX_VERIFY_NUDGES


def coding_verify_guidance(config: Optional[dict[str, Any]] = None) -> Optional[str]:
    """Return the optional guidance appended to verification-stop nudges."""
    if not is_truthy_value(_agent_cfg(config).get("verify_guidance", True), default=True):
        return None
    return CODING_VERIFY_GUIDANCE


def _agent_cfg(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
    agent_cfg = config.get("agent") if isinstance(config, dict) else None
    return agent_cfg if isinstance(agent_cfg, dict) else {}


__all__ = ["CODING_VERIFY_GUIDANCE", "DEFAULT_MAX_VERIFY_NUDGES", "coding_verify_guidance", "max_verify_nudges"]
