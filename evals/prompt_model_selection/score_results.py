"""Validate and score recorded prompt/model A/B results without model calls.

The scorer binds every row to a content-derived study, arm, route, prompt,
case, and fixture revision. Each metric retains its evidence basis, so
measured and estimated values are never blended.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Any, Iterable


TOKEN_BASES = {"provider_usage", "tokenizer_estimate"}
LATENCY_BASES = {"wall_clock", "estimate"}
COST_BASES = {"provider_reported", "invoice", "catalog_estimate", "unknown"}
QUALITY_BASES = {"deterministic", "blind_human"}
COMMIT_REVISION_PATTERN = re.compile(r"^git-commit-sha1:([0-9a-f]{40})$")
BLOB_REVISION_PATTERN = re.compile(r"^git-blob-sha1:([0-9a-f]{40})$")
DEFAULT_MIN_REPETITIONS = 3
ARMS_SCHEMA_VERSION = 3


def _revision(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: each row must be an object")
        rows.append(row)
    return rows


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("every case requires a non-empty case_id")
        if case_id in cases:
            raise ValueError(f"duplicate case_id: {case_id}")
        route_id = row.get("route_id")
        if not isinstance(route_id, str) or not route_id:
            raise ValueError(f"{case_id}: route_id must be a non-empty string")
        fixture = row.get("fixture")
        if not isinstance(fixture, str) or not fixture:
            raise ValueError(f"{case_id}: fixture must be a non-empty string")
        checks = row.get("quality_checks")
        critical = row.get("critical_checks")
        if not isinstance(checks, dict) or not checks:
            raise ValueError(f"{case_id}: quality_checks must be a non-empty object")
        if not isinstance(critical, list) or not set(critical).issubset(checks):
            raise ValueError(f"{case_id}: critical_checks must name quality_checks")

        clean = dict(row)
        clean["fixture_revision"] = _revision(fixture)
        clean["case_revision"] = _revision(clean)
        cases[case_id] = clean
    if not cases:
        raise ValueError("cases manifest must contain at least one case")
    return cases


def _git(
    repository: Path, *args: str, error_context: str
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "--literal-pathspecs",
                "-C",
                str(repository),
                *args,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(f"{error_context}: could not execute Git: {exc}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Git command failed"
        raise ValueError(f"{error_context}: {detail}")
    return result


def _repository_root(manifest_path: Path) -> Path:
    resolved_manifest = manifest_path.resolve()
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "--literal-pathspecs",
                "-C",
                str(resolved_manifest.parent),
                "rev-parse",
                "--show-toplevel",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ValueError(
            f"{manifest_path}: could not execute Git while resolving repository: {exc}"
        ) from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError(f"{manifest_path}: must be inside a Git working tree")
    repository = Path(result.stdout.strip()).resolve()
    try:
        resolved_manifest.relative_to(repository)
    except ValueError as exc:
        raise ValueError(
            f"{manifest_path}: must resolve inside its Git working tree"
        ) from exc
    return repository


def _source_commit(arm_id: str, raw_arm: dict[str, Any], repository: Path) -> str:
    source_commit = raw_arm.get("source_commit")
    match = (
        COMMIT_REVISION_PATTERN.fullmatch(source_commit)
        if isinstance(source_commit, str)
        else None
    )
    if match is None:
        raise ValueError(f"{arm_id}: source_commit must use git-commit-sha1:<40 hex>")
    commit_id = match.group(1)
    object_type = _git(
        repository,
        "cat-file",
        "-t",
        commit_id,
        error_context=f"{arm_id}: source commit {commit_id} is unavailable",
    ).stdout.strip()
    if object_type != "commit":
        raise ValueError(
            f"{arm_id}: source_commit must identify a commit, got {object_type!r}"
        )
    return source_commit


def _prompt_source_path(arm_id: str, route_id: str, source: Any) -> str:
    if not isinstance(source, str) or not source:
        raise ValueError(f"{arm_id}/{route_id}: prompt source paths must be non-empty")
    normalized = PurePosixPath(source)
    if (
        normalized.is_absolute()
        or source != normalized.as_posix()
        or ".." in normalized.parts
        or "\\" in source
        or "\x00" in source
    ):
        raise ValueError(
            f"{arm_id}/{route_id}: prompt source must be a normalized "
            f"repository-relative path: {source!r}"
        )
    return source


def _verify_prompt_sources(
    repository: Path,
    arm_id: str,
    source_commit: str,
    prompt_sources: dict[str, str],
) -> None:
    commit_id = source_commit.removeprefix("git-commit-sha1:")
    for source, declared_revision in sorted(prompt_sources.items()):
        result = _git(
            repository,
            "ls-tree",
            "-z",
            "--full-tree",
            commit_id,
            "--",
            source,
            error_context=f"{arm_id}: cannot resolve {source} at {commit_id}",
        )
        records = [record for record in result.stdout.split("\0") if record]
        if len(records) != 1 or "\t" not in records[0]:
            raise ValueError(
                f"{arm_id}: {source} is missing or not a file at source commit "
                f"{commit_id}"
            )
        metadata, resolved_source = records[0].split("\t", 1)
        fields = metadata.split()
        if len(fields) != 3:
            raise ValueError(
                f"{arm_id}: Git returned invalid metadata for {source} at {commit_id}"
            )
        mode, object_type, object_id = fields
        if (
            resolved_source != source
            or object_type != "blob"
            or mode not in {"100644", "100755"}
        ):
            raise ValueError(
                f"{arm_id}: {source} must resolve to a regular Git blob at "
                f"source commit {commit_id}"
            )
        declared_id = declared_revision.removeprefix("git-blob-sha1:")
        if object_id != declared_id:
            raise ValueError(
                f"{arm_id}: declared blob {declared_id} does not match {source} "
                f"at source commit {commit_id}; Git resolves {object_id}"
            )


def load_arms(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ValueError("arms manifest must be an object")
    if raw.get("schema_version") != ARMS_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {ARMS_SCHEMA_VERSION} for repository-bound provenance"
        )
    repository = _repository_root(path)
    study_id = raw.get("study_id")
    if not isinstance(study_id, str) or not study_id:
        raise ValueError("study_id must be a non-empty string")
    default_min = raw.get("default_min_repetitions", DEFAULT_MIN_REPETITIONS)
    if (
        isinstance(default_min, bool)
        or not isinstance(default_min, int)
        or default_min < 1
    ):
        raise ValueError("default_min_repetitions must be a positive integer")
    raw_arms = raw.get("arms")
    if not isinstance(raw_arms, dict) or len(raw_arms) < 2:
        raise ValueError("arms must contain at least two named configurations")

    arms: dict[str, dict[str, Any]] = {}
    for arm_id, raw_arm in raw_arms.items():
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("every arm requires a non-empty identifier")
        if not isinstance(raw_arm, dict):
            raise ValueError(f"{arm_id}: arm must be an object")
        source_commit = _source_commit(arm_id, raw_arm, repository)
        raw_routes = raw_arm.get("routes")
        if not isinstance(raw_routes, dict) or not raw_routes:
            raise ValueError(f"{arm_id}: routes must be a non-empty object")

        routes: dict[str, dict[str, Any]] = {}
        arm_prompt_sources: dict[str, str] = {}
        for route_id, raw_route in raw_routes.items():
            if not isinstance(route_id, str) or not route_id:
                raise ValueError(
                    f"{arm_id}: every route requires a non-empty identifier"
                )
            if not isinstance(raw_route, dict):
                raise ValueError(f"{arm_id}/{route_id}: route must be an object")
            route = dict(raw_route)
            for field in ("provider", "model"):
                if not isinstance(route.get(field), str) or not route[field]:
                    raise ValueError(
                        f"{arm_id}/{route_id}: {field} must be a concrete non-empty string"
                    )
                if route[field].startswith("inherit"):
                    raise ValueError(
                        f"{arm_id}/{route_id}: {field} cannot use inherited routing"
                    )
            prompt_sources = route.get("prompt_sources")
            if not isinstance(prompt_sources, dict) or not prompt_sources:
                raise ValueError(
                    f"{arm_id}/{route_id}: prompt_sources must be a non-empty object"
                )
            for raw_source, source_revision in prompt_sources.items():
                source = _prompt_source_path(arm_id, route_id, raw_source)
                if (
                    not isinstance(source_revision, str)
                    or BLOB_REVISION_PATTERN.fullmatch(source_revision) is None
                ):
                    raise ValueError(
                        f"{arm_id}/{route_id}: {source} must use git-blob-sha1:<40 hex>"
                    )
                previous = arm_prompt_sources.setdefault(source, source_revision)
                if previous != source_revision:
                    raise ValueError(
                        f"{arm_id}: {source} has conflicting prompt-source revisions"
                    )
            route["prompt_revision"] = _revision({
                "source_commit": source_commit,
                "prompt_sources": prompt_sources,
            })
            route["route_revision"] = _revision({
                "source_commit": source_commit,
                "route": route,
            })
            routes[route_id] = route

        _verify_prompt_sources(repository, arm_id, source_commit, arm_prompt_sources)
        arm = dict(raw_arm)
        arm["source_commit"] = source_commit
        arm["routes"] = routes
        arm["arm_revision"] = _revision(arm)
        arms[arm_id] = arm

    study = dict(raw)
    study["default_min_repetitions"] = default_min
    study["arms"] = arms
    return study


def _validate_contract(study: dict[str, Any], cases: dict[str, dict[str, Any]]) -> None:
    route_ids = {case["route_id"] for case in cases.values()}
    for arm_id, arm in study["arms"].items():
        missing = route_ids - set(arm["routes"])
        if missing:
            raise ValueError(f"{arm_id}: missing case routes {sorted(missing)}")


def evaluation_revision(study: dict[str, Any], cases: dict[str, dict[str, Any]]) -> str:
    _validate_contract(study, cases)
    return _revision({"study": study, "cases": cases})


def expected_provenance(
    study: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    arm: str,
    case_id: str,
) -> dict[str, str]:
    revision = evaluation_revision(study, cases)
    if arm not in study["arms"]:
        raise ValueError(f"unknown arm: {arm!r}")
    if case_id not in cases:
        raise ValueError(f"unknown case_id: {case_id!r}")
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
    }


def _nonnegative_number(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{field} must be a non-negative number")
    return float(value)


def validate_result(
    row: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    study: dict[str, Any],
) -> dict[str, Any]:
    arm = row.get("arm")
    case_id = row.get("case_id")
    if arm not in study["arms"]:
        raise ValueError(f"unknown arm: {arm!r}")
    if case_id not in cases:
        raise ValueError(f"unknown case_id: {case_id!r}")

    expected = expected_provenance(study, cases, arm, case_id)
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise ValueError(
                f"{arm}/{case_id}: {field} does not match the evaluation manifest"
            )

    repetition = row.get("repetition")
    if (
        isinstance(repetition, bool)
        or not isinstance(repetition, int)
        or repetition < 1
    ):
        raise ValueError("repetition must be a positive integer")

    case = cases[case_id]
    checks = row.get("checks")
    expected_checks = set(case["quality_checks"])
    if not isinstance(checks, dict) or set(checks) != expected_checks:
        raise ValueError(
            f"{case_id}: checks must exactly match {sorted(expected_checks)}"
        )
    if any(not isinstance(value, bool) for value in checks.values()):
        raise ValueError(f"{case_id}: every check result must be boolean")

    bases = {
        "quality_basis": QUALITY_BASES,
        "tokens_basis": TOKEN_BASES,
        "latency_basis": LATENCY_BASES,
        "cost_basis": COST_BASES,
    }
    for field, allowed in bases.items():
        if row.get(field) not in allowed:
            raise ValueError(f"{field} must be one of {sorted(allowed)}")

    clean = dict(row)
    for field in ("input_tokens", "output_tokens", "latency_ms", "cost_usd"):
        clean[field] = _nonnegative_number(row, field)
    quality_score = sum(checks.values()) / len(checks)
    critical_pass = all(checks[name] for name in case["critical_checks"])
    clean["quality_score"] = quality_score
    clean["quality_pass"] = critical_pass and quality_score >= float(
        case.get("min_quality", 1.0)
    )
    return clean


def _validate_matrix(
    rows: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    study: dict[str, Any],
    min_repetitions: int,
) -> int:
    seen: set[tuple[str, str, int]] = set()
    repetitions: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        key = (row["arm"], row["case_id"], row["repetition"])
        if key in seen:
            raise ValueError(
                "duplicate result for "
                f"arm={row['arm']!r}, case_id={row['case_id']!r}, "
                f"repetition={row['repetition']}"
            )
        seen.add(key)
        repetitions[(row["arm"], row["case_id"])].add(row["repetition"])

    cells = [repetitions[(arm, case_id)] for arm in study["arms"] for case_id in cases]
    first = cells[0]
    if any(cell != first for cell in cells[1:]):
        raise ValueError(
            "incomplete or unpaired result matrix: every arm/case must contain "
            "the same repetition identifiers"
        )
    if len(first) < min_repetitions:
        raise ValueError(
            f"every arm/case requires at least {min_repetitions} distinct repetitions"
        )
    expected = set(range(1, len(first) + 1))
    if first != expected:
        raise ValueError(
            "repetition identifiers must be contiguous integers starting at 1"
        )
    return len(first)


def _numeric_summary(values: Iterable[float]) -> dict[str, Any]:
    materialized = list(values)
    return {
        "n": len(materialized),
        "mean": mean(materialized),
        "median": median(materialized),
        "total": sum(materialized),
    }


def _summaries_by_basis(
    rows: list[dict[str, Any]], basis_field: str, value_field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[basis_field]].append(row[value_field])
    return {
        basis: _numeric_summary(values) for basis, values in sorted(grouped.items())
    }


def summarize(
    rows: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    study: dict[str, Any],
    *,
    min_repetitions: int | None = None,
) -> dict[str, Any]:
    _validate_contract(study, cases)
    required_repetitions = (
        study["default_min_repetitions"] if min_repetitions is None else min_repetitions
    )
    if (
        isinstance(required_repetitions, bool)
        or not isinstance(required_repetitions, int)
        or required_repetitions < 1
    ):
        raise ValueError("min_repetitions must be a positive integer")

    validated = [validate_result(row, cases, study) for row in rows]
    repetition_count = _validate_matrix(validated, cases, study, required_repetitions)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in validated:
        row["total_tokens"] = row["input_tokens"] + row["output_tokens"]
        by_arm[row["arm"]].append(row)

    arms: dict[str, dict[str, Any]] = {}
    for arm, arm_rows in sorted(by_arm.items()):
        arms[arm] = {
            "arm_revision": study["arms"][arm]["arm_revision"],
            "source_commit": study["arms"][arm]["source_commit"],
            "runs": len(arm_rows),
            "distinct_cases": len({row["case_id"] for row in arm_rows}),
            "quality_score_by_basis": _summaries_by_basis(
                arm_rows, "quality_basis", "quality_score"
            ),
            "quality_pass_rate_by_basis": {
                basis: _numeric_summary(
                    float(row["quality_pass"])
                    for row in arm_rows
                    if row["quality_basis"] == basis
                )
                for basis in sorted({row["quality_basis"] for row in arm_rows})
            },
            "total_tokens_by_basis": _summaries_by_basis(
                arm_rows, "tokens_basis", "total_tokens"
            ),
            "latency_ms_by_basis": _summaries_by_basis(
                arm_rows, "latency_basis", "latency_ms"
            ),
            "cost_usd_by_basis": _summaries_by_basis(
                arm_rows, "cost_basis", "cost_usd"
            ),
        }

    arm_count = len(study["arms"])
    case_count = len(cases)
    return {
        "study_id": study["study_id"],
        "evaluation_revision": evaluation_revision(study, cases),
        "result_count": len(validated),
        "case_count": case_count,
        "matrix": {
            "arms": arm_count,
            "cases": case_count,
            "repetitions_per_arm_case": repetition_count,
            "paired_runs": case_count * repetition_count,
        },
        "arms": arms,
        "interpretation": {
            "measured_bases": {
                "tokens": ["provider_usage"],
                "latency": ["wall_clock"],
                "cost": ["provider_reported", "invoice"],
            },
            "estimated_bases": {
                "tokens": ["tokenizer_estimate"],
                "latency": ["estimate"],
                "cost": ["catalog_estimate"],
            },
            "rule": "Compare arms only within the same metric basis; do not blend measured and estimated values.",
        },
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help="Recorded result JSONL; this tool never invokes a model",
    )
    parser.add_argument("--cases", type=Path, default=here / "cases.jsonl")
    parser.add_argument("--arms", type=Path, default=here / "arms.json")
    parser.add_argument(
        "--min-repetitions",
        type=int,
        help="Required repetitions per arm/case (default: arms manifest, normally 3)",
    )
    args = parser.parse_args()
    cases = load_cases(args.cases)
    study = load_arms(args.arms)
    report = summarize(
        load_jsonl(args.results),
        cases,
        study,
        min_repetitions=args.min_repetitions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
