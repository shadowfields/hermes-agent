from copy import deepcopy
from pathlib import Path

import pytest

from evals.prompt_model_selection.score_results import (
    evaluation_revision,
    load_arms,
    load_cases,
    summarize,
)


EVAL_DIR = Path(__file__).parents[2] / "evals" / "prompt_model_selection"


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


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("evaluation_revision", "sha256:" + "0" * 64),
        ("case_revision", "sha256:" + "1" * 64),
        ("fixture_revision", "sha256:" + "2" * 64),
        ("arm_revision", "sha256:" + "3" * 64),
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
