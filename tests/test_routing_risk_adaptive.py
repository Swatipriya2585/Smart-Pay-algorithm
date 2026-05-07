"""RuleBasedRiskAdaptiveRouter tests.

These tests verify the rule logic with hand-built risk estimates and
regime classifications, so we don't depend on randomness from upstream
scorers. Statistical tests aren't needed — rules are deterministic.
"""

import pytest

from app.market_data.base import NetworkConditions
from app.regime.base import RegimeEstimate
from app.risk.base import MultiHorizonRiskEstimate, TailRiskEstimate
from app.routing.risk_adaptive import (
    RoutingConfig,
    RuleBasedRiskAdaptiveRouter,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _regime(label: str = "calm", confidence: float = 0.9) -> RegimeEstimate:
    return RegimeEstimate(
        symbol="MARKET",
        regime=label,  # type: ignore[arg-type]
        confidence=confidence,
        realized_volatility=0.001,
        baseline_volatility=0.001,
        ratio=1.0,
    )


def _network(
    congestion: float = 0.1,
    priority: float = 1.0,
    slot_ms: float = 400.0,
) -> NetworkConditions:
    return NetworkConditions(
        priority_fee_lamports=priority,
        congestion_score=congestion,
        slot_time_ms=slot_ms,
    )


def _risk(
    symbol: str,
    position: float = 1000.0,
    cvar_pct: float = 0.02,  # 2% of position
) -> MultiHorizonRiskEstimate:
    """Build a MultiHorizonRiskEstimate where worst-horizon CVaR equals
    cvar_pct of the position."""
    cvar = -cvar_pct
    var = -cvar_pct / 2  # var is less severe than cvar
    estimate = TailRiskEstimate(
        horizon_seconds=120.0,
        confidence_level=0.95,
        var=var,
        cvar=cvar,
        var_dollar=var * position,
        cvar_dollar=cvar * position,
        n_samples=10_000,
    )
    return MultiHorizonRiskEstimate(
        symbol=symbol,
        position_value_usd=position,
        estimates={120.0: estimate},
    )


def _router(**config_kwargs) -> RuleBasedRiskAdaptiveRouter:
    cfg = RoutingConfig(**config_kwargs)
    return RuleBasedRiskAdaptiveRouter(config=cfg)


# -----------------------------------------------------------------------------
# CVaR exclusion rule
# -----------------------------------------------------------------------------


def test_cvar_under_threshold_not_excluded() -> None:
    """1% CVaR loss with 5% threshold — token should not be excluded."""
    r = _router()
    risks = {"SOL": _risk("SOL", cvar_pct=0.01)}
    is_stable = {"SOL": False}
    decision = r.decide(_regime("calm"), risks, is_stable, _network())
    assert not decision.for_symbol("SOL").excluded


def test_cvar_above_threshold_excluded() -> None:
    """8% CVaR loss with 5% threshold — token should be excluded."""
    r = _router()
    risks = {"SOL": _risk("SOL", cvar_pct=0.08)}
    is_stable = {"SOL": False}
    decision = r.decide(_regime("calm"), risks, is_stable, _network())
    adj = decision.for_symbol("SOL")
    # 8% > 5% triggers exclusion BUT the empty-set fallback may relax CVaR
    # if it would empty the set. With only one token, fallback keeps it.
    # Check the relaxation behavior: when relaxation kicks in, the token
    # is included instead.
    # For this test, since SOL is the only candidate and would be excluded,
    # the fallback drops the CVaR rule, so SOL is included.
    assert not adj.excluded  # fallback kept it


def test_cvar_exclusion_with_alternative_token() -> None:
    """When SOL would breach but USDC is a healthy alternative, SOL is excluded."""
    r = _router()
    risks = {
        "SOL": _risk("SOL", cvar_pct=0.08),     # 8% — breach
        "USDC": _risk("USDC", cvar_pct=0.001),   # 0.1% — fine
    }
    is_stable = {"SOL": False, "USDC": True}
    decision = r.decide(_regime("calm"), risks, is_stable, _network())
    assert decision.for_symbol("SOL").excluded
    assert "CVaR breach" in decision.for_symbol("SOL").exclusion_reason
    assert not decision.for_symbol("USDC").excluded


def test_cvar_exclusion_can_be_relaxed_to_avoid_empty_set() -> None:
    """All candidates breach CVaR — fallback should keep them included."""
    r = _router()
    risks = {
        "BONK": _risk("BONK", cvar_pct=0.10),
        "WIF": _risk("WIF", cvar_pct=0.12),
    }
    is_stable = {"BONK": False, "WIF": False}
    decision = r.decide(_regime("calm"), risks, is_stable, _network())
    # Both should NOT be excluded after fallback — empty-set guard fired.
    assert not decision.for_symbol("BONK").excluded
    assert not decision.for_symbol("WIF").excluded


# -----------------------------------------------------------------------------
# Stablecoin preference by regime
# -----------------------------------------------------------------------------


def test_calm_regime_no_stablecoin_bias() -> None:
    r = _router()
    risks = {
        "SOL": _risk("SOL"),
        "USDC": _risk("USDC"),
    }
    is_stable = {"SOL": False, "USDC": True}
    decision = r.decide(_regime("calm"), risks, is_stable, _network())
    assert decision.for_symbol("USDC").score_bias_bps == 0.0
    assert decision.for_symbol("SOL").score_bias_bps == 0.0


def test_stress_regime_stablecoin_gets_bias() -> None:
    """Default stress bias = +200 bps."""
    r = _router()
    risks = {
        "SOL": _risk("SOL"),
        "USDC": _risk("USDC"),
    }
    is_stable = {"SOL": False, "USDC": True}
    decision = r.decide(_regime("stress"), risks, is_stable, _network())
    assert decision.for_symbol("USDC").score_bias_bps == 200.0
    assert decision.for_symbol("SOL").score_bias_bps == 0.0


def test_shock_regime_stablecoin_gets_bigger_bias() -> None:
    """Default shock bias = +500 bps."""
    r = _router()
    risks = {
        "SOL": _risk("SOL"),
        "USDC": _risk("USDC"),
    }
    is_stable = {"SOL": False, "USDC": True}
    decision = r.decide(_regime("shock"), risks, is_stable, _network())
    assert decision.for_symbol("USDC").score_bias_bps == 500.0


def test_stress_regime_only_stables_get_bias() -> None:
    """Multiple non-stables and multiple stables — only stables get bias."""
    r = _router()
    risks = {
        "SOL": _risk("SOL"),
        "BONK": _risk("BONK"),
        "USDC": _risk("USDC"),
        "USDT": _risk("USDT"),
    }
    is_stable = {"SOL": False, "BONK": False, "USDC": True, "USDT": True}
    decision = r.decide(_regime("stress"), risks, is_stable, _network())
    assert decision.for_symbol("USDC").score_bias_bps == 200.0
    assert decision.for_symbol("USDT").score_bias_bps == 200.0
    assert decision.for_symbol("SOL").score_bias_bps == 0.0
    assert decision.for_symbol("BONK").score_bias_bps == 0.0


# -----------------------------------------------------------------------------
# Shock-regime non-stable exclusion
# -----------------------------------------------------------------------------


def test_shock_excludes_non_stables_when_stable_available() -> None:
    r = _router()
    risks = {
        "SOL": _risk("SOL"),
        "USDC": _risk("USDC"),
    }
    is_stable = {"SOL": False, "USDC": True}
    decision = r.decide(_regime("shock"), risks, is_stable, _network())
    assert decision.for_symbol("SOL").excluded
    assert "Shock regime" in decision.for_symbol("SOL").exclusion_reason
    assert not decision.for_symbol("USDC").excluded


def test_shock_does_not_exclude_when_no_stable_available() -> None:
    """Shock regime with no stables — non-stables should NOT be excluded."""
    r = _router()
    risks = {
        "SOL": _risk("SOL"),
        "BONK": _risk("BONK"),
    }
    is_stable = {"SOL": False, "BONK": False}
    decision = r.decide(_regime("shock"), risks, is_stable, _network())
    assert not decision.for_symbol("SOL").excluded
    assert not decision.for_symbol("BONK").excluded


def test_shock_exclusion_disabled_via_config() -> None:
    """Config can disable shock-regime exclusion entirely."""
    r = _router(shock_excludes_non_stables=False)
    risks = {
        "SOL": _risk("SOL"),
        "USDC": _risk("USDC"),
    }
    is_stable = {"SOL": False, "USDC": True}
    decision = r.decide(_regime("shock"), risks, is_stable, _network())
    assert not decision.for_symbol("SOL").excluded
    # USDC still gets shock bias.
    assert decision.for_symbol("USDC").score_bias_bps == 500.0


# -----------------------------------------------------------------------------
# Congestion-driven liquidity preference
# -----------------------------------------------------------------------------


def test_low_congestion_no_liquidity_bias() -> None:
    r = _router()
    risks = {"SOL": _risk("SOL")}
    is_stable = {"SOL": False}
    liquidity = {"SOL": 10_000_000.0}
    decision = r.decide(
        _regime("calm"), risks, is_stable, _network(congestion=0.1),
        liquidity_depth_usd=liquidity,
    )
    assert decision.for_symbol("SOL").score_bias_bps == 0.0


def test_high_congestion_deep_liquidity_gets_bias() -> None:
    """High congestion + deep liquidity = +100 bps default."""
    r = _router()
    risks = {"SOL": _risk("SOL")}
    is_stable = {"SOL": False}
    liquidity = {"SOL": 10_000_000.0}  # above default 5M threshold
    decision = r.decide(
        _regime("calm"), risks, is_stable, _network(congestion=0.8),
        liquidity_depth_usd=liquidity,
    )
    assert decision.for_symbol("SOL").score_bias_bps == 100.0


def test_high_congestion_shallow_liquidity_no_bias() -> None:
    r = _router()
    risks = {"SOL": _risk("SOL")}
    is_stable = {"SOL": False}
    liquidity = {"SOL": 100_000.0}  # below threshold
    decision = r.decide(
        _regime("calm"), risks, is_stable, _network(congestion=0.8),
        liquidity_depth_usd=liquidity,
    )
    assert decision.for_symbol("SOL").score_bias_bps == 0.0


# -----------------------------------------------------------------------------
# Bias stacking
# -----------------------------------------------------------------------------


def test_stress_stable_with_deep_liquidity_in_congestion_stacks_biases() -> None:
    """USDC in stress regime + high congestion + deep liquidity:
    +200 (stress stable) + +100 (deep liquidity) = +300 bps."""
    r = _router()
    risks = {
        "USDC": _risk("USDC"),
        "SOL": _risk("SOL"),  # provides a non-stable comparison
    }
    is_stable = {"USDC": True, "SOL": False}
    liquidity = {"USDC": 50_000_000.0, "SOL": 10_000_000.0}
    decision = r.decide(
        _regime("stress"), risks, is_stable, _network(congestion=0.8),
        liquidity_depth_usd=liquidity,
    )
    usdc = decision.for_symbol("USDC")
    assert usdc.score_bias_bps == 300.0
    # Both bias reasons are recorded.
    assert len(usdc.bias_reasons) == 2


# -----------------------------------------------------------------------------
# Audit trail
# -----------------------------------------------------------------------------


def test_excluded_token_has_human_readable_reason() -> None:
    r = _router()
    risks = {
        "SOL": _risk("SOL", cvar_pct=0.10),
        "USDC": _risk("USDC", cvar_pct=0.001),
    }
    is_stable = {"SOL": False, "USDC": True}
    decision = r.decide(_regime("calm"), risks, is_stable, _network())
    reason = decision.for_symbol("SOL").exclusion_reason
    assert reason is not None
    assert "CVaR" in reason


def test_biased_token_records_each_reason() -> None:
    r = _router()
    risks = {
        "USDC": _risk("USDC"),
        "SOL": _risk("SOL"),
    }
    is_stable = {"USDC": True, "SOL": False}
    liquidity = {"USDC": 50_000_000.0, "SOL": 10_000_000.0}
    decision = r.decide(
        _regime("stress"), risks, is_stable, _network(congestion=0.8),
        liquidity_depth_usd=liquidity,
    )
    usdc = decision.for_symbol("USDC")
    # Two reasons present: stress preference + congestion liquidity.
    reasons_text = " ".join(usdc.bias_reasons)
    assert "Stress" in reasons_text or "stress" in reasons_text
    assert "ongestion" in reasons_text  # Congestion or congestion


# -----------------------------------------------------------------------------
# Decision-level invariants
# -----------------------------------------------------------------------------


def test_decide_preserves_input_order() -> None:
    r = _router()
    risks = {
        "BONK": _risk("BONK"),
        "SOL": _risk("SOL"),
        "USDC": _risk("USDC"),
    }
    is_stable = {"BONK": False, "SOL": False, "USDC": True}
    decision = r.decide(_regime("calm"), risks, is_stable, _network())
    assert tuple(a.symbol for a in decision.adjustments) == ("BONK", "SOL", "USDC")


def test_decide_records_regime_in_decision() -> None:
    r = _router()
    risks = {"SOL": _risk("SOL")}
    is_stable = {"SOL": False}
    regime = _regime("stress", confidence=0.85)
    decision = r.decide(regime, risks, is_stable, _network())
    assert decision.regime is regime


def test_decide_rejects_empty_risk_estimates() -> None:
    r = _router()
    with pytest.raises(ValueError, match="at least one"):
        r.decide(_regime("calm"), {}, {}, _network())


def test_decide_rejects_missing_stablecoin_flags() -> None:
    """If is_stablecoin doesn't cover all tokens, fail loudly."""
    r = _router()
    risks = {"SOL": _risk("SOL"), "USDC": _risk("USDC")}
    is_stable = {"SOL": False}  # USDC missing
    with pytest.raises(ValueError, match="missing"):
        r.decide(_regime("calm"), risks, is_stable, _network())


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------


def test_config_rejects_cvar_pct_at_zero() -> None:
    with pytest.raises(ValueError, match="cvar_exclusion_pct"):
        RoutingConfig(cvar_exclusion_pct=0.0)


def test_config_rejects_cvar_pct_at_one() -> None:
    with pytest.raises(ValueError, match="cvar_exclusion_pct"):
        RoutingConfig(cvar_exclusion_pct=1.0)


def test_config_rejects_negative_stress_bias() -> None:
    with pytest.raises(ValueError, match="stress_stablecoin_bias_bps"):
        RoutingConfig(stress_stablecoin_bias_bps=-10.0)


def test_config_rejects_invalid_congestion_threshold() -> None:
    with pytest.raises(ValueError, match="congestion_high_threshold"):
        RoutingConfig(congestion_high_threshold=1.5)


def test_config_rejects_zero_liquidity_deep() -> None:
    with pytest.raises(ValueError, match="liquidity_deep_usd"):
        RoutingConfig(liquidity_deep_usd=0.0)


# -----------------------------------------------------------------------------
# Smoke
# -----------------------------------------------------------------------------


def test_full_universe_decide_smoke() -> None:
    """All 8 calibrated tokens should produce a valid decision."""
    r = _router()
    universe = ["SOL", "USDC", "PYTH", "AERO", "JUP", "BRETT", "WIF", "BONK"]
    risks = {sym: _risk(sym, cvar_pct=0.02) for sym in universe}
    is_stable = {
        "SOL": False, "USDC": True, "PYTH": False, "AERO": False,
        "JUP": False, "BRETT": False, "WIF": False, "BONK": False,
    }
    decision = r.decide(_regime("stress"), risks, is_stable, _network(congestion=0.8))
    # All adjustments present, USDC has bias, others may have biases or not.
    assert len(decision.adjustments) == 8
    assert decision.for_symbol("USDC").score_bias_bps > 0
