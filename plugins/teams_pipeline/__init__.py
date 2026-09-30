"""Teams meeting pipeline plugin.

Registers operator-facing CLI surfaces and the pipeline's owned auxiliary LLM
route. It adds no model tools.
"""

from __future__ import annotations

from plugins.teams_pipeline.cli import register_cli, teams_pipeline_command


def register(ctx) -> None:
    ctx.register_auxiliary_task(
        key="teams_summary",
        display_name="Teams summary",
        description="Grounded Microsoft Teams meeting summary extraction",
        defaults={"timeout": 120, "reasoning_effort": ""},
    )
    ctx.register_cli_command(
        name="teams-pipeline",
        help="Inspect and operate the Microsoft Teams meeting pipeline",
        setup_fn=register_cli,
        handler_fn=teams_pipeline_command,
        description=(
            "Operator CLI for the Microsoft Teams meeting pipeline. "
            "Lists jobs, inspects stored runs, replays jobs, validates Graph "
            "setup, and maintains Graph subscriptions."
        ),
    )
