from copy import deepcopy
import json
from pathlib import Path
import subprocess

import pytest

from evals.prompt_model_selection.score_results import (
    evaluation_revision,
    load_arms,
    load_cases,
    summarize,
)


EVAL_DIR = Path(__file__).parents[2] / "evals" / "prompt_model_selection"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def provenance_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "prompt-eval@example.invalid")
    _git(repo, "config", "user.name", "Prompt Eval Test")

    prompts = repo / "prompts"
    prompts.mkdir()
    source = prompts / "system.py"
    unrelated = prompts / "unrelated.py"
    source.write_text('PROMPT = "base"\n', encoding="utf-8")
    unrelated.write_text('NOT_THE_PROMPT = "unrelated"\n', encoding="utf-8")
    _git(repo, "add", "prompts/system.py", "prompts/unrelated.py")
    _git(repo, "commit", "--quiet", "-m", "base prompt")
    base_commit = _git(repo, "rev-parse", "HEAD")
    base_blob = _git(repo, "rev-parse", f"{base_commit}:prompts/system.py")

    source.write_text('PROMPT = "candidate"\n', encoding="utf-8")
    _git(repo, "add", "prompts/system.py")
    _git(repo, "commit", "--quiet", "-m", "candidate prompt")
    candidate_commit = _git(repo, "rev-parse", "HEAD")
    candidate_blob = _git(repo, "rev-parse", f"{candidate_commit}:prompts/system.py")
    unrelated_blob = _git(repo, "rev-parse", f"{candidate_commit}:prompts/unrelated.py")
    prompts_tree = _git(repo, "rev-parse", f"{candidate_commit}:prompts")

    manifest = {
        "schema_version": 3,
        "study_id": "test-provenance",
        "default_min_repetitions": 1,
        "arms": {
            "current": {
                "source_commit": f"git-commit-sha1:{base_commit}",
                "routes": {
                    "task": {
                        "provider": "test",
                        "model": "base",
                        "prompt_sources": {
                            "prompts/system.py": f"git-blob-sha1:{base_blob}"
                        },
                    }
                },
            },
            "proposed": {
                "source_commit": f"git-commit-sha1:{candidate_commit}",
                "routes": {
                    "task": {
                        "provider": "test",
                        "model": "candidate",
                        "prompt_sources": {
                            "prompts/system.py": f"git-blob-sha1:{candidate_blob}"
                        },
                    }
                },
            },
        },
    }
    manifest_path = repo / "evals" / "arms.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {
        "repo": repo,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "base_commit": base_commit,
        "base_blob": base_blob,
        "candidate_commit": candidate_commit,
        "candidate_blob": candidate_blob,
        "unrelated_blob": unrelated_blob,
        "prompts_tree": prompts_tree,
    }


def _write_manifest(provenance_repo, manifest):
    provenance_repo["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")


def _contract():
    cases = load_cases(EVAL_DIR / "cases.jsonl")
    study = load_arms(EVAL_DIR / "arms.json")
    revision = evaluation_revision(study, cases)
    return study, cases, revision


def _row(*, study, cases, revision, arm, case_id, repetition):
    case = cases[case_id]
    arm_config = study["arms"][arm]
    route = arm_config["routes"][case["route_id"]]
    return {
        "study_id": study["study_id"],
        "evaluation_revision": revision,
        "case_id": case_id,
        "case_revision": case["case_revision"],
        "fixture_revision": case["fixture_revision"],
        "arm": arm,
        "arm_revision": arm_config["arm_revision"],
        "source_commit": arm_config["source_commit"],
        "route_id": case["route_id"],
        "route_revision": route["route_revision"],
        "provider": route["provider"],
        "model": route["model"],
        "prompt_revision": route["prompt_revision"],
        "repetition": repetition,
        "checks": {name: True for name in case["quality_checks"]},
        "quality_basis": "deterministic",
        "input_tokens": 100,
        "output_tokens": 20,
        "tokens_basis": "provider_usage",
        "latency_ms": 250,
        "latency_basis": "wall_clock",
        "cost_usd": 0.01,
        "cost_basis": "provider_reported",
    }


def _complete_rows(*, repetitions=3):
    study, cases, revision = _contract()
    rows = [
        _row(
            study=study,
            cases=cases,
            revision=revision,
            arm=arm,
            case_id=case_id,
            repetition=repetition,
        )
        for arm in study["arms"]
        for case_id in cases
        for repetition in range(1, repetitions + 1)
    ]
    return study, cases, rows


def test_summary_accepts_complete_balanced_matrix_and_separates_evidence_bases():
    study, cases, rows = _complete_rows()
    estimated = rows[-1]
    estimated["tokens_basis"] = "tokenizer_estimate"
    estimated["latency_basis"] = "estimate"
    estimated["cost_basis"] = "catalog_estimate"

    report = summarize(rows, cases, study)

    assert report["matrix"] == {
        "arms": 2,
        "cases": 8,
        "repetitions_per_arm_case": 3,
        "paired_runs": 24,
    }
    proposed = report["arms"]["proposed"]
    assert set(proposed["total_tokens_by_basis"]) == {
        "provider_usage",
        "tokenizer_estimate",
    }
    assert set(proposed["latency_ms_by_basis"]) == {"estimate", "wall_clock"}
    assert set(proposed["cost_usd_by_basis"]) == {
        "catalog_estimate",
        "provider_reported",
    }
    assert report["interpretation"]["rule"].startswith("Compare arms only within")


def test_shipped_arms_resolve_to_exact_git_commits_and_prompt_blobs():
    study = load_arms(EVAL_DIR / "arms.json")

    assert study["arms"]["current"]["source_commit"] == (
        "git-commit-sha1:4546cd40ecb764dd265c38923b227b572b355d32"
    )
    assert study["arms"]["proposed"]["source_commit"] == (
        "git-commit-sha1:a5b9586e654ea7c70d5d85714b8bd4a4eb01ed75"
    )


def test_load_arms_accepts_repository_bound_provenance(provenance_repo):
    study = load_arms(provenance_repo["manifest_path"])

    assert study["arms"]["current"]["source_commit"].endswith(
        provenance_repo["base_commit"]
    )
    assert study["arms"]["proposed"]["source_commit"].endswith(
        provenance_repo["candidate_commit"]
    )


def test_load_arms_rejects_stale_blob_from_another_commit(provenance_repo):
    manifest = deepcopy(provenance_repo["manifest"])
    manifest["arms"]["current"]["routes"]["task"]["prompt_sources"] = {
        "prompts/system.py": f"git-blob-sha1:{provenance_repo['candidate_blob']}"
    }
    _write_manifest(provenance_repo, manifest)

    with pytest.raises(ValueError, match="does not match prompts/system.py"):
        load_arms(provenance_repo["manifest_path"])


def test_load_arms_rejects_blob_from_unrelated_path(provenance_repo):
    manifest = deepcopy(provenance_repo["manifest"])
    manifest["arms"]["proposed"]["routes"]["task"]["prompt_sources"] = {
        "prompts/system.py": f"git-blob-sha1:{provenance_repo['unrelated_blob']}"
    }
    _write_manifest(provenance_repo, manifest)

    with pytest.raises(ValueError, match="does not match prompts/system.py"):
        load_arms(provenance_repo["manifest_path"])


def test_load_arms_rejects_prompt_blob_from_wrong_source_commit(provenance_repo):
    manifest = deepcopy(provenance_repo["manifest"])
    manifest["arms"]["current"]["source_commit"] = (
        f"git-commit-sha1:{provenance_repo['candidate_commit']}"
    )
    _write_manifest(provenance_repo, manifest)

    with pytest.raises(ValueError, match="does not match prompts/system.py"):
        load_arms(provenance_repo["manifest_path"])


def test_load_arms_rejects_manifest_outside_a_git_repository(provenance_repo, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    manifest_path = outside / "arms.json"
    manifest_path.write_text(json.dumps(provenance_repo["manifest"]), encoding="utf-8")

    with pytest.raises(ValueError, match="must be inside a Git working tree"):
        load_arms(manifest_path)


def test_load_arms_rejects_symbolic_source_revision(provenance_repo):
    manifest = deepcopy(provenance_repo["manifest"])
    manifest["arms"]["current"]["source_commit"] = "HEAD"
    _write_manifest(provenance_repo, manifest)

    with pytest.raises(ValueError, match="git-commit-sha1"):
        load_arms(provenance_repo["manifest_path"])


def test_load_arms_rejects_blob_object_as_source_commit(provenance_repo):
    manifest = deepcopy(provenance_repo["manifest"])
    manifest["arms"]["current"]["source_commit"] = (
        f"git-commit-sha1:{provenance_repo['base_blob']}"
    )
    _write_manifest(provenance_repo, manifest)

    with pytest.raises(ValueError, match="must identify a commit"):
        load_arms(provenance_repo["manifest_path"])


def test_load_arms_rejects_tree_as_prompt_source(provenance_repo):
    manifest = deepcopy(provenance_repo["manifest"])
    manifest["arms"]["proposed"]["routes"]["task"]["prompt_sources"] = {
        "prompts": f"git-blob-sha1:{provenance_repo['prompts_tree']}"
    }
    _write_manifest(provenance_repo, manifest)

    with pytest.raises(ValueError, match="must resolve to a regular Git blob"):
        load_arms(provenance_repo["manifest_path"])


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("evaluation_revision", "sha256:" + "0" * 64),
        ("case_revision", "sha256:" + "1" * 64),
        ("fixture_revision", "sha256:" + "2" * 64),
        ("arm_revision", "sha256:" + "3" * 64),
        ("source_commit", "git-commit-sha1:" + "6" * 40),
        ("route_id", "wrong-route"),
        ("route_revision", "sha256:" + "4" * 64),
        ("provider", "wrong-provider"),
        ("model", "wrong-model"),
        ("prompt_revision", "sha256:" + "5" * 64),
    ],
)
def test_summary_rejects_results_that_do_not_match_manifest(field, bad_value):
    study, cases, rows = _complete_rows()
    rows[0][field] = bad_value

    with pytest.raises(ValueError, match=field):
        summarize(rows, cases, study)


def test_summary_rejects_missing_unpaired_result():
    study, cases, rows = _complete_rows()
    rows.pop()

    with pytest.raises(ValueError, match="incomplete or unpaired result matrix"):
        summarize(rows, cases, study)


def test_summary_rejects_incomplete_result_row():
    study, cases, rows = _complete_rows()
    del rows[0]["model"]

    with pytest.raises(ValueError, match="model"):
        summarize(rows, cases, study)


def test_summary_rejects_duplicate_arm_case_repetition():
    study, cases, rows = _complete_rows()
    rows.append(deepcopy(rows[0]))

    with pytest.raises(ValueError, match="duplicate result"):
        summarize(rows, cases, study)


def test_default_requires_three_distinct_repetitions_per_arm_case():
    study, cases, rows = _complete_rows(repetitions=2)

    with pytest.raises(ValueError, match="at least 3 distinct repetitions"):
        summarize(rows, cases, study)

    report = summarize(rows, cases, study, min_repetitions=2)
    assert report["matrix"]["repetitions_per_arm_case"] == 2


@pytest.mark.parametrize("bad_repetition", [True, 0, -1, 1.5, "1"])
def test_summary_rejects_invalid_repetition_identifiers(bad_repetition):
    study, cases, rows = _complete_rows()
    rows[0]["repetition"] = bad_repetition

    with pytest.raises(ValueError, match="repetition"):
        summarize(rows, cases, study)


def test_summary_rejects_noncontiguous_repetition_set():
    study, cases, rows = _complete_rows()
    for row in rows:
        if row["repetition"] == 3:
            row["repetition"] = 4

    with pytest.raises(ValueError, match="contiguous"):
        summarize(rows, cases, study)
