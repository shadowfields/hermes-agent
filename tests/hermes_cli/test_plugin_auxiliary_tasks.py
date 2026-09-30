"""Plugin-registered auxiliary tasks merge into the built-in task list and ``_reset_aux_to_auto``."""

from __future__ import annotations

import json

import pytest

from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
    get_plugin_auxiliary_tasks,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def patched_manager(monkeypatch):
    """Replace the module-level singleton with a fresh manager for the test.

    Restored automatically after the test by monkeypatch.
    """
    from hermes_cli import plugins as plugins_mod

    fresh = PluginManager()
    fresh._discovered = True
    monkeypatch.setattr(plugins_mod, "_PLUGIN_MANAGER", fresh, raising=False)

    def _stub_get_manager() -> PluginManager:
        return fresh

    monkeypatch.setattr(plugins_mod, "get_plugin_manager", _stub_get_manager)
    monkeypatch.setattr(plugins_mod, "_ensure_plugins_discovered", _stub_get_manager)
    yield fresh


# ── _all_aux_tasks merges built-in + plugin ──────────────────────────────────


def test_all_aux_tasks_includes_plugin_registered(patched_manager):
    from hermes_cli.main_provider_setup import _AUX_TASKS, _all_aux_tasks

    manifest = PluginManifest(name="hindsight")
    ctx = PluginContext(manifest, patched_manager)
    ctx.register_auxiliary_task(
        key="memory_retain_filter",
        display_name="Memory retain filter",
        description="hindsight pre-retain dedup/extract",
    )

    merged = _all_aux_tasks()
    keys = [k for k, _, _ in merged]
    # Built-ins preserved (and come first)
    builtin_keys = [k for k, _, _ in _AUX_TASKS]
    assert keys[: len(builtin_keys)] == builtin_keys
    # Plugin task appended
    assert "memory_retain_filter" in keys
    plugin_entry = next(t for t in merged if t[0] == "memory_retain_filter")
    assert plugin_entry == (
        "memory_retain_filter",
        "Memory retain filter",
        "hindsight pre-retain dedup/extract",
    )


# ── _reset_aux_to_auto includes plugin tasks ─────────────────────────────────


def test_reset_aux_to_auto_resets_plugin_tasks(tmp_path, monkeypatch, patched_manager):
    """Plugin task with non-auto config gets reset alongside built-ins."""
    from pathlib import Path
    from hermes_cli.config import load_config, save_config
    from hermes_cli.main_provider_setup import _reset_aux_to_auto

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir(exist_ok=True)

    manifest = PluginManifest(name="plug")
    ctx = PluginContext(manifest, patched_manager)
    ctx.register_auxiliary_task(
        key="my_aux",
        display_name="My Aux",
        description="d",
    )

    # Manually configure the plugin task to non-auto
    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    aux["my_aux"] = {"provider": "openrouter", "model": "gpt-4o", "base_url": "", "api_key": ""}
    save_config(cfg)

    n = _reset_aux_to_auto()
    assert n >= 1

    cfg = load_config()
    assert cfg["auxiliary"]["my_aux"]["provider"] == "auto"
    assert cfg["auxiliary"]["my_aux"]["model"] == ""


# ── auxiliary_client._get_auxiliary_task_config defaults layering ────────────


def test_teams_summary_real_discovery_drives_config_and_dashboard_routes(tmp_path, monkeypatch):
    """An enabled bundled plugin owns its route from discovery through config surfaces."""
    from agent.auxiliary_client import _get_auxiliary_task_config
    from hermes_cli.config import load_config
    from hermes_cli.main_provider_setup import _all_aux_tasks
    from hermes_cli.plugins import discover_plugins
    from hermes_cli.web_routers.models import get_auxiliary_models
    from hermes_cli.web_server_config import _apply_aux_assignment_sync

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        json.dumps(
            {
                "plugins": {"enabled": ["teams_pipeline"]},
                "auxiliary": {
                    "teams_summary": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-sonnet-5",
                        "timeout": 45,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    discover_plugins(force=True)

    registered = {entry["key"]: entry for entry in get_plugin_auxiliary_tasks()}
    assert registered["teams_summary"]["plugin"] == "teams_pipeline"
    assert registered["teams_summary"]["defaults"]["timeout"] == 120
    assert "teams_summary" in {key for key, _name, _description in _all_aux_tasks()}

    resolved = _get_auxiliary_task_config("teams_summary")
    assert resolved["provider"] == "openrouter"
    assert resolved["model"] == "anthropic/claude-sonnet-5"
    assert resolved["timeout"] == 45
    assert resolved["reasoning_effort"] == ""

    dashboard = get_auxiliary_models()
    dashboard_task = next(task for task in dashboard["tasks"] if task["task"] == "teams_summary")
    assert dashboard_task["display_name"] == "Teams summary"
    assert dashboard_task["description"]

    cfg = load_config()
    result = _apply_aux_assignment_sync(
        cfg,
        "openai",
        "gpt-5-mini",
        "teams_summary",
        "",
        "",
        reasoning_effort="low",
    )
    assert result["tasks"] == ["teams_summary"]
    persisted = _get_auxiliary_task_config("teams_summary")
    assert persisted["provider"] == "openai"
    assert persisted["model"] == "gpt-5-mini"
    assert persisted["reasoning_effort"] == "low"


def test_disabled_teams_plugin_does_not_leave_a_core_auxiliary_route(tmp_path, monkeypatch):
    from agent.auxiliary_client import _get_auxiliary_task_config
    from hermes_cli.plugins import discover_plugins

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        json.dumps({"plugins": {"disabled": ["teams_pipeline"]}}),
        encoding="utf-8",
    )

    discover_plugins(force=True)

    assert "teams_summary" not in {entry["key"] for entry in get_plugin_auxiliary_tasks()}
    assert _get_auxiliary_task_config("teams_summary") == {}
