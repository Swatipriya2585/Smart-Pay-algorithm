"""SolanaCostScorer tests.

Slippage and gas math are closed-form — tested with hand-computed expected
values to high precision (1e-9). Settlement risk involves the forecaster
output, so its tolerances are looser (~5%) but the math turning forecast
into risk is itself exact and testable.
"""

import math

import numpy as np
import pytest

from app.cost.scorer import (
    DEFAULT_BASE_FEE_LAMPORTS,
    DEFAULT_COMPUTE_UNITS_PER_SWAP,
    SOL_LAMPORTS,
    SolanaCostConfig,
    SolanaCostScorer,
)
from app.forecasting.base import HorizonForecast, MultiHorizonForecast
from app.forecasting.garch import GARCHForecaster
from app.market_data.base import NetworkConditions, PricePath, TokenMarketData
from app.market_data.calibration import Calibration
from app.market_data.mock import MockConfig, MockMarketData


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _calm_network() -> NetworkConditions:
    return NetworkConditions(
        priority_fee_lamports=1.0,
        congestion_score=0.1,
        slot_time_ms=400.0,
    )


def _shock_network() -> NetworkConditions:
    return NetworkConditions(
        priority_fee_lamports=100_000.0,
        congestion_score=0.95,
        slot_time_ms=400.0,
    )


def _synthetic_forecast(
    symbol: str = "TEST",
    vol_per_5s: float = 0.001,
    horizons: tuple[float, ...] = (5.0, 30.0, 120.0),
) -> MultiHorizonForecast:
    """Forecast where each horizon's vol scales as sqrt(time) from the 5s value."""
    forecasts = {}
    for h in horizons:
        scale = math.sqrt(h / 5.0)
        v = vol_per_5s * scale
        forecasts[h] = HorizonForecast(
            horizon_seconds=h,
            predicted_return=0.0,
            predicted_volatility=v,
            confidence_lower_95=-2 * v,
            confidence_upper_95=2 * v,
        )
    return MultiHorizonForecast(symbol=symbol, horizons=forecasts)


def _synthetic_token(
    symbol: str = "TEST",
    liquidity_depth_usd: float = 5_000_000.0,
) -> TokenMarketData:
    return TokenMarketData(
        symbol=symbol,
        mint="testmint",
        path=PricePath(
            symbol=symbol,
            prices_usd=np.array([100.0] * 100),
            interval_seconds=60.0,
        ),
        liquidity_depth_usd=liquidity_depth_usd,
        spread_bps=5.0,
    )


def _real_forecast(symbol: str) -> MultiHorizonForecast:
    f = GARCHForecaster(calibration=Calibration())
    mock = MockMarketData(config=MockConfig(n_observations=1440, seed=7))
    return f.forecast(mock.fetch([symbol])[0])


def _real_token(symbol: str) -> TokenMarketData:
    mock = MockMarketData(config=MockConfig(n_observations=1440, seed=7))
    return mock.fetch([symbol])[0]


# -----------------------------------------------------------------------------
# Slippage math (closed form, hand-computed)
# -----------------------------------------------------------------------------


def test_slippage_formula_at_known_values() -> None:
    """$1000 swap into a $5M depth pool. Expected: -1000 * 1000/(1000+5_000_000)."""
    expected = -1000.0 * (1000.0 / (1000.0 + 5_000_000.0))
    actual = SolanaCostScorer._compute_slippage(
        position_value_usd=1000.0, liquidity_depth_usd=5_000_000.0
    )
    assert abs(actual - expected) < 1e-12


def test_slippage_zero_position_returns_zero() -> None:
    assert (
        SolanaCostScorer._compute_slippage(
            position_value_usd=0.0, liquidity_depth_usd=5_000_000.0
        )
        == 0.0
    )


def test_slippage_zero_liquidity_returns_full_position_loss() -> None:
    """Pathological case: empty pool. Conservative: 100% loss."""
    assert (
        SolanaCostScorer._compute_slippage(
            position_value_usd=1000.0, liquidity_depth_usd=0.0
        )
        == -1000.0
    )


def test_slippage_deeper_pool_costs_less() -> None:
    """Strict monotone: deeper pool = less slippage."""
    shallow = SolanaCostScorer._compute_slippage(
        position_value_usd=1000.0, liquidity_depth_usd=20_000.0
    )
    deep = SolanaCostScorer._compute_slippage(
        position_value_usd=1000.0, liquidity_depth_usd=10_000_000.0
    )
    # Both negative; deep is closer to zero.
    assert deep > shallow


def test_slippage_larger_position_costs_more() -> None:
    small = SolanaCostScorer._compute_slippage(
        position_value_usd=100.0, liquidity_depth_usd=1_000_000.0
    )
    large = SolanaCostScorer._compute_slippage(
        position_value_usd=10_000.0, liquidity_depth_usd=1_000_000.0
    )
    # Both negative; large is more negative.
    assert large < small


# -----------------------------------------------------------------------------
# Gas math (closed form, hand-computed)
# -----------------------------------------------------------------------------


def test_gas_formula_calm_network() -> None:
    """Calm: priority_fee=1, CU=200K, base=5000 -> 205000 lamports."""
    scorer = SolanaCostScorer()
    gas = scorer._compute_gas(_calm_network())
    expected_lamports = DEFAULT_BASE_FEE_LAMPORTS + 1.0 * DEFAULT_COMPUTE_UNITS_PER_SWAP
    expected_sol = expected_lamports / SOL_LAMPORTS
    expected_dollar = -expected_sol * 150.0  # default sol_price_usd
    assert abs(gas - expected_dollar) < 1e-12


def test_gas_formula_shock_network() -> None:
    """Shock: priority_fee=100K -> 5000 + 100K * 200K = 20,000,005,000 lamports."""
    scorer = SolanaCostScorer()
    gas = scorer._compute_gas(_shock_network())
    expected_lamports = DEFAULT_BASE_FEE_LAMPORTS + 100_000.0 * DEFAULT_COMPUTE_UNITS_PER_SWAP
    expected_sol = expected_lamports / SOL_LAMPORTS
    expected_dollar = -expected_sol * 150.0
    assert abs(gas - expected_dollar) < 1e-12


def test_gas_higher_priority_fee_costs_more() -> None:
    scorer = SolanaCostScorer()
    calm_gas = scorer._compute_gas(_calm_network())
    shock_gas = scorer._compute_gas(_shock_network())
    # Both negative; shock more negative.
    assert shock_gas < calm_gas


def test_gas_with_custom_sol_price() -> None:
    """sol_price_usd should scale gas linearly."""
    scorer_a = SolanaCostScorer(SolanaCostConfig(sol_price_usd=100.0))
    scorer_b = SolanaCostScorer(SolanaCostConfig(sol_price_usd=200.0))
    a = scorer_a._compute_gas(_calm_network())
    b = scorer_b._compute_gas(_calm_network())
    # 2x sol price -> 2x dollar gas (negative).
    assert abs(b - 2 * a) < 1e-12


# -----------------------------------------------------------------------------
# Settlement seconds (linear in congestion)
# -----------------------------------------------------------------------------


def test_settlement_seconds_calm() -> None:
    scorer = SolanaCostScorer()
    # congestion=0.1: mult = 1 + (5-1)*0.1 = 1.4. slot=400ms = 0.4s. total = 0.56s.
    s = scorer._compute_settlement_seconds(_calm_network())
    assert abs(s - 0.56) < 1e-12


def test_settlement_seconds_shock() -> None:
    scorer = SolanaCostScorer()
    # congestion=0.95: mult = 1 + (5-1)*0.95 = 4.8. slot=0.4s. total = 1.92s.
    s = scorer._compute_settlement_seconds(_shock_network())
    assert abs(s - 1.92) < 1e-12


def test_settlement_seconds_zero_congestion_equals_slot_time() -> None:
    scorer = SolanaCostScorer()
    network = NetworkConditions(
        priority_fee_lamports=0.0, congestion_score=0.0, slot_time_ms=400.0
    )
    s = scorer._compute_settlement_seconds(network)
    assert abs(s - 0.4) < 1e-12


# -----------------------------------------------------------------------------
# Settlement risk (forecast-linked, exact math)
# -----------------------------------------------------------------------------


def test_settlement_risk_known_inputs() -> None:
    """vol_per_5s=0.001 -> per_second=0.001/sqrt(5).

    settlement_seconds=0.5 -> settlement_vol = (0.001/sqrt(5)) * sqrt(0.5).
    Position $1000 -> risk = -1000 * settlement_vol.
    """
    scorer = SolanaCostScorer()
    forecast = _synthetic_forecast(vol_per_5s=0.001)
    risk = scorer._compute_settlement_risk(
        forecast=forecast,
        settlement_seconds=0.5,
        position_value_usd=1000.0,
    )
    per_second = 0.001 / math.sqrt(5.0)
    expected = -1000.0 * per_second * math.sqrt(0.5)
    assert abs(risk - expected) < 1e-12


def test_settlement_risk_zero_position() -> None:
    scorer = SolanaCostScorer()
    forecast = _synthetic_forecast()
    assert scorer._compute_settlement_risk(
        forecast=forecast, settlement_seconds=0.5, position_value_usd=0.0
    ) == 0.0


def test_settlement_risk_scales_with_sqrt_settlement_time() -> None:
    """4x longer settlement -> 2x more risk (sqrt scaling)."""
    scorer = SolanaCostScorer()
    forecast = _synthetic_forecast(vol_per_5s=0.001)
    risk_short = scorer._compute_settlement_risk(
        forecast=forecast, settlement_seconds=0.5, position_value_usd=1000.0
    )
    risk_long = scorer._compute_settlement_risk(
        forecast=forecast, settlement_seconds=2.0, position_value_usd=1000.0
    )
    # 2.0/0.5 = 4 -> sqrt(4) = 2x
    assert abs(risk_long / risk_short - 2.0) < 1e-9


# -----------------------------------------------------------------------------
# End-to-end estimate() shape
# -----------------------------------------------------------------------------


def test_estimate_returns_multihorizon_with_all_horizons() -> None:
    scorer = SolanaCostScorer()
    out = scorer.estimate(
        data=_synthetic_token(),
        forecast=_synthetic_forecast(),
        network=_calm_network(),
        position_value_usd=1000.0,
    )
    assert sorted(out.breakdowns.keys()) == [5.0, 30.0, 120.0]
    assert out.symbol == "TEST"


def test_estimate_total_equals_sum_of_components() -> None:
    """Schema enforces this, but we test the producer respects it."""
    scorer = SolanaCostScorer()
    out = scorer.estimate(
        data=_synthetic_token(),
        forecast=_synthetic_forecast(),
        network=_calm_network(),
        position_value_usd=1000.0,
    )
    for br in out.breakdowns.values():
        component_sum = (
            br.slippage_dollar + br.gas_dollar + br.settlement_risk_dollar
        )
        assert abs(br.total_cost_dollar - component_sum) < 1e-9


def test_estimate_components_identical_across_horizons() -> None:
    """Slippage/gas/settlement are horizon-agnostic in the v1 scorer."""
    scorer = SolanaCostScorer()
    out = scorer.estimate(
        data=_synthetic_token(),
        forecast=_synthetic_forecast(),
        network=_calm_network(),
        position_value_usd=1000.0,
    )
    horizons = out.horizon_seconds_list()
    base = out.at(horizons[0])
    for h in horizons[1:]:
        other = out.at(h)
        assert other.slippage_dollar == base.slippage_dollar
        assert other.gas_dollar == base.gas_dollar
        assert other.settlement_risk_dollar == base.settlement_risk_dollar


def test_estimate_negative_position_rejected() -> None:
    scorer = SolanaCostScorer()
    with pytest.raises(ValueError, match="position_value_usd"):
        scorer.estimate(
            data=_synthetic_token(),
            forecast=_synthetic_forecast(),
            network=_calm_network(),
            position_value_usd=-100.0,
        )


def test_estimate_zero_position_yields_zero_costs() -> None:
    scorer = SolanaCostScorer()
    out = scorer.estimate(
        data=_synthetic_token(),
        forecast=_synthetic_forecast(),
        network=_calm_network(),
        position_value_usd=0.0,
    )
    for br in out.breakdowns.values():
        # Gas is fixed, doesn't scale with position. Slippage and settlement
        # do scale with position. Position=0 -> slippage=0, settlement=0,
        # gas still has its fixed value.
        assert br.slippage_dollar == 0.0
        assert br.settlement_risk_dollar == 0.0
        assert br.gas_dollar < 0  # gas is fixed cost


# -----------------------------------------------------------------------------
# Cross-token / cross-network sanity
# -----------------------------------------------------------------------------


def test_shallow_pool_token_costs_more_than_deep_pool() -> None:
    """Same payment, two tokens with very different liquidity depth."""
    scorer = SolanaCostScorer()
    deep = scorer.estimate(
        data=_synthetic_token(symbol="DEEP", liquidity_depth_usd=10_000_000.0),
        forecast=_synthetic_forecast(),
        network=_calm_network(),
        position_value_usd=1000.0,
    )
    shallow = scorer.estimate(
        data=_synthetic_token(symbol="SHAL", liquidity_depth_usd=10_000.0),
        forecast=_synthetic_forecast(),
        network=_calm_network(),
        position_value_usd=1000.0,
    )
    # worst_total_cost for shallow should be more negative than for deep.
    assert shallow.worst_total_cost_dollar() < deep.worst_total_cost_dollar()


def test_shock_network_costs_more_than_calm() -> None:
    scorer = SolanaCostScorer()
    calm = scorer.estimate(
        data=_synthetic_token(),
        forecast=_synthetic_forecast(),
        network=_calm_network(),
        position_value_usd=1000.0,
    )
    shock = scorer.estimate(
        data=_synthetic_token(),
        forecast=_synthetic_forecast(),
        network=_shock_network(),
        position_value_usd=1000.0,
    )
    assert shock.worst_total_cost_dollar() < calm.worst_total_cost_dollar()


# -----------------------------------------------------------------------------
# Config validation
# -----------------------------------------------------------------------------


def test_config_rejects_negative_base_fee() -> None:
    with pytest.raises(ValueError, match="base_fee_lamports"):
        SolanaCostConfig(base_fee_lamports=-100.0)


def test_config_rejects_zero_compute_units() -> None:
    with pytest.raises(ValueError, match="compute_units_per_swap"):
        SolanaCostConfig(compute_units_per_swap=0.0)


def test_config_rejects_zero_sol_price() -> None:
    with pytest.raises(ValueError, match="sol_price_usd"):
        SolanaCostConfig(sol_price_usd=0.0)


def test_config_rejects_below_one_congestion_multiplier() -> None:
    """Congestion can't speed things up — multiplier must be >= 1."""
    with pytest.raises(ValueError, match="congestion"):
        SolanaCostConfig(max_congestion_settlement_multiplier=0.5)


# -----------------------------------------------------------------------------
# End-to-end smoke with real GARCH forecasts
# -----------------------------------------------------------------------------


def test_full_universe_estimate_smoke() -> None:
    """All 8 calibrated tokens should produce valid cost estimates."""
    scorer = SolanaCostScorer()
    network = _calm_network()
    universe = ["SOL", "USDC", "PYTH", "AERO", "JUP", "BRETT", "WIF", "BONK"]
    for sym in universe:
        out = scorer.estimate(
            data=_real_token(sym),
            forecast=_real_forecast(sym),
            network=network,
            position_value_usd=1000.0,
        )
        for br in out.breakdowns.values():
            assert math.isfinite(br.total_cost_dollar)
            assert br.total_cost_dollar <= 0


def test_volatile_token_has_more_settlement_risk_than_stable() -> None:
    """BONK (high vol) should have more settlement risk than USDC (stable)."""
    scorer = SolanaCostScorer()
    network = _calm_network()
    bonk_out = scorer.estimate(
        data=_real_token("BONK"),
        forecast=_real_forecast("BONK"),
        network=network,
        position_value_usd=1000.0,
    )
    usdc_out = scorer.estimate(
        data=_real_token("USDC"),
        forecast=_real_forecast("USDC"),
        network=network,
        position_value_usd=1000.0,
    )
    # Settlement risk is more negative (more severe) for BONK than USDC.
    assert bonk_out.at(5.0).settlement_risk_dollar < usdc_out.at(5.0).settlement_risk_dollar
