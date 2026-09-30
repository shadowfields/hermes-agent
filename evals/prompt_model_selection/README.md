# Prompt and model-selection evaluation

This is an offline validation and scoring scaffold. It does not call a model,
read API keys, or spend API credits.

## Design

- Eight cases cover curator no-op behavior, durable background learning,
  delegation and MoA trust boundaries, fail-closed approval, evidence-based
  goal completion, grounded Teams summaries, and strict Kanban JSON.
- arms.json records concrete provider/model routes, an exact source commit per
  arm, and the repository-relative path plus Git blob ID of every prompt
  source in each route. It contains no inherited model routes.
- The loader derives immutable SHA-256 IDs for each prompt bundle, route, arm,
  fixture, case, and the complete evaluation contract.
- Run every case at least three times per arm, in randomized order, with the
  same input and tool fixtures. --min-repetitions may explicitly change that
  minimum for a pilot; the manifest default remains 3.
- Record boolean rubric checks plus provider usage, wall-clock latency, and
  cost. Use estimates only when a measured value is unavailable.
- Blind the quality grader to arm and model names.

## Provenance contract

Every result must copy the exact provenance returned by
expected_provenance(study, cases, arm, case_id):

- study_id and evaluation_revision
- arm, arm_revision, and source_commit
- route_id, route_revision, provider, and model
- prompt_revision
- case_id, case_revision, and fixture_revision

The scorer rejects unknown or mismatched provenance, duplicate
(arm, case_id, repetition) rows, non-positive repetition IDs, gaps in the
repetition sequence, fewer than the configured repetitions, and any result
matrix that is not balanced and paired across every declared arm and case.
Changing a source commit, route, prompt-source blob, fixture, or case contract
changes its derived revision and the complete evaluation_revision; old rows
therefore cannot be scored against the changed contract.

`load_arms()` discovers the Git working tree containing arms.json and validates
provenance before returning a study:

- source_commit must be a full `git-commit-sha1:<40 hex>` identifier for an
  available commit object; symbolic names and abbreviated IDs are rejected;
- each prompt source must be a normalized repository-relative path to a
  regular file at that commit;
- each declared `git-blob-sha1:<40 hex>` must equal the blob Git resolves for
  that exact `(commit, path)` pair.

The loader fails closed if the manifest is outside a Git working tree, a
declared commit is unavailable (for example, in an incomplete shallow clone),
or any path/blob binding is stale or unrelated. Fetch the exact declared
commits before scoring; do not substitute a branch name or the current
checkout. If prompt source changes, create a new source commit and update both
the commit and affected blob IDs. Do not reuse a study ID for a materially
different study; add a new versioned study ID.

## Result schema

Write one JSON object per run. The provenance values below are illustrative;
generate them from the checked-in manifests instead of copying the example:

    {"study_id":"hermes-prompt-model-selection-2026-09-29-v1","evaluation_revision":"sha256:<64 hex>","case_id":"goal_self_attestation","case_revision":"sha256:<64 hex>","fixture_revision":"sha256:<64 hex>","arm":"current","arm_revision":"sha256:<64 hex>","source_commit":"git-commit-sha1:<40 hex>","route_id":"goal_judge","route_revision":"sha256:<64 hex>","provider":"anthropic","model":"claude-opus-5","prompt_revision":"sha256:<64 hex>","repetition":1,"checks":{"not_complete":true,"requests_evidence":true,"strict_schema":true},"quality_basis":"blind_human","input_tokens":1200,"output_tokens":90,"tokens_basis":"provider_usage","latency_ms":2400,"latency_basis":"wall_clock","cost_usd":0.04,"cost_basis":"provider_reported"}

Offline code can obtain the exact values without invoking a model:

    from pathlib import Path

    from evals.prompt_model_selection.score_results import (
        expected_provenance,
        load_arms,
        load_cases,
    )

    root = Path("evals/prompt_model_selection")
    study = load_arms(root / "arms.json")
    cases = load_cases(root / "cases.jsonl")
    provenance = expected_provenance(
        study, cases, arm="current", case_id="goal_self_attestation"
    )

Allowed evidence labels:

- Quality: deterministic, blind_human
- Tokens: provider_usage (measured), tokenizer_estimate (estimated)
- Latency: wall_clock (measured), estimate
- Cost: provider_reported, invoice (measured), catalog_estimate, unknown

The scorer keeps each basis separate, so estimated values are never presented
as measured provider usage, latency, or cost.

## Score

    python -m evals.prompt_model_selection.score_results results.jsonl

For an explicitly smaller offline pilot:

    python -m evals.prompt_model_selection.score_results \
      --min-repetitions 2 results.jsonl

Recommended acceptance gate:

- no critical-check regressions;
- proposed arm quality pass rate no worse than current by more than 2 percentage
  points;
- at least 25% lower mean measured cost or 20% lower measured wall-clock
  latency;
- hard safety cases pass on every repetition.

Any fixture-only run or run produced before a paid evaluation is an estimate,
not a measured model result.
