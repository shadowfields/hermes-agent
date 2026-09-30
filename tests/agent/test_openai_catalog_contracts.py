"""Cross-surface contracts for current OpenAI reasoning models.

These tests intentionally derive models and rates from production registries.
They protect joins and estimator behavior without copying a vendor catalog into
the test suite.
"""

from decimal import Decimal
from unittest.mock import patch

from agent.model_metadata import get_model_context_length
from agent.models_dev import get_model_capabilities
from agent.reasoning_effort import EFFORT_LADDER, codex_supported_efforts
from agent.usage_pricing import (
    CanonicalUsage,
    PricingEntry,
    _lookup_official_docs_pricing,
    estimate_usage_cost,
    resolve_billing_route,
)
from hermes_cli.models_catalog_static import _PROVIDER_MODELS


_ONE_MILLION = Decimal(1_000_000)


def test_current_openai_family_joins_catalog_metadata_efforts_and_pricing():
    catalog = set(_PROVIDER_MODELS["openai-api"])
    catalog_family = {model for model in catalog if model.startswith("gpt-6")}

    assert catalog_family
    assert not any(model.startswith("gpt-6-astra") for model in catalog)

    with patch("agent.models_dev.fetch_models_dev", return_value={}):
        capabilities_by_model = {
            model: get_model_capabilities("openai-api", model)
            for model in sorted(catalog_family)
        }

        for model, capabilities in capabilities_by_model.items():
            assert capabilities is not None, model
            assert capabilities.supports_tools is True, model
            assert capabilities.supports_vision is True, model
            assert capabilities.supports_reasoning is True, model
            assert capabilities.model_family == "gpt-6", model
            assert capabilities.context_window > capabilities.max_output_tokens > 0, model
            assert capabilities.context_window == get_model_context_length(
                model, provider="openai-api",
            ), model

            efforts = codex_supported_efforts(model)
            assert efforts, model
            assert set(efforts) <= set(EFFORT_LADDER), model
            assert list(efforts) == sorted(efforts, key=EFFORT_LADDER.index), model

            route = resolve_billing_route(model, provider="openai-api")
            pricing = _lookup_official_docs_pricing(route)
            assert route.provider == "openai", model
            assert pricing is not None, model
            assert pricing.source == "official_docs_snapshot", model
            assert pricing.source_url and pricing.pricing_version, model

        for model, capabilities in capabilities_by_model.items():
            if not model.endswith("-pro"):
                continue
            base = model.removesuffix("-pro")
            assert base in capabilities_by_model, model
            assert capabilities == capabilities_by_model[base], model

        assert get_model_capabilities("openai-api", "gpt-6-unknown-pro") is None


def _tiered_openai_catalog_rows() -> list[tuple[str, PricingEntry]]:
    rows = []
    for model in _PROVIDER_MODELS["openai-api"]:
        route = resolve_billing_route(model, provider="openai-api")
        entry = _lookup_official_docs_pricing(route)
        if entry is not None and entry.tier_threshold_tokens is not None:
            rows.append((model, entry))
    return rows


def _expected_token_cost(
    usage: CanonicalUsage, entry: PricingEntry, *, above: bool,
) -> Decimal:
    rates = (
        entry.input_cost_per_million_above if above else entry.input_cost_per_million,
        entry.output_cost_per_million_above if above else entry.output_cost_per_million,
        entry.cache_read_cost_per_million_above if above else entry.cache_read_cost_per_million,
        entry.cache_write_cost_per_million_above if above else entry.cache_write_cost_per_million,
    )
    assert all(rate is not None for rate in rates)
    return sum(
        Decimal(tokens) * rate
        for tokens, rate in zip(
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.cache_write_tokens,
            ),
            rates,
            strict=True,
        )
    ) / _ONE_MILLION


def test_tiered_openai_estimator_uses_each_catalog_entry_rates():
    rows = _tiered_openai_catalog_rows()
    assert rows

    for model, entry in rows:
        threshold = entry.tier_threshold_tokens
        assert threshold is not None and threshold > 2, model

        input_tokens = threshold // 3
        cache_read_tokens = threshold // 3
        cache_write_tokens = threshold - input_tokens - cache_read_tokens
        at_threshold = CanonicalUsage(
            input_tokens=input_tokens,
            output_tokens=max(1, threshold // 100),
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        above_threshold = CanonicalUsage(
            input_tokens=at_threshold.input_tokens,
            output_tokens=at_threshold.output_tokens,
            cache_read_tokens=at_threshold.cache_read_tokens,
            cache_write_tokens=at_threshold.cache_write_tokens + 1,
        )

        assert at_threshold.prompt_tokens == threshold
        assert above_threshold.prompt_tokens == threshold + 1

        base_result = estimate_usage_cost(model, at_threshold, provider="openai-api")
        above_result = estimate_usage_cost(model, above_threshold, provider="openai-api")

        assert base_result.amount_usd == _expected_token_cost(
            at_threshold, entry, above=False,
        ), model
        assert above_result.amount_usd == _expected_token_cost(
            above_threshold, entry, above=True,
        ), model
        assert base_result.pricing_version == entry.pricing_version, model
        assert above_result.pricing_version == entry.pricing_version, model
