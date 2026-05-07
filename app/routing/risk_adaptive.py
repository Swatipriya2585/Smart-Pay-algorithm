"""
Rule-based risk-adaptive router.

Implements the RiskAdaptiveRouter Protocol with three rule families that
translate regime classification + CVaR estimates + network state into
per-token routing adjustments (excludes and biases).

Rule families:

1. CVaR exclusion (always active, regime-independent):
   If a token's worst-horizon CVaR exceeds cvar_exclusion_pct of the
   payment value, exclude it entirely. Default 5% — aggressive.
   Rationale: any token where the average bad-case loss exceeds 5% of
   the payment is too risky to spend, regardless of how cheap it looks.

2. Regime-driven stablecoin preference:
   - In CALM: no bias.
   - In STRESS: +stress_stablecoin_bias_bps for stablecoins.
   - In SHOCK: +shock_stablecoin_bias_bps for stablecoins, AND exclude
     all non-stablecoins IF at least one stablecoin is in the candidate
     set. Falls back to relaxing this rule if it would leave zero
     candidates.

3. Congestion-driven liquidity preference:
   When network congestion is high (>= congestion_high_threshold),
   tokens with deep liquidity (>= liquidity_deep_usd) get a bias.
   Rationale: under congestion, deep-liquidity tokens are more likely
   to settle without partial fills.

Empty-set fallback:
If applying all rules would leave zero candidates, the router
progressively relaxes:
  Step 1: drop the shock-regime non-stable exclusion (keep CVaR exclusion).
  Step 2: drop the CVaR exclusion too (keep only flagged tokens, no exclusions).
This ensures the bandit always has at least one candidate to choose from.

Aggressive defaults rationale:
For a payment system, the cost of routing through a tail-risk token is
much greater than the cost of paying slightly more in slippage. We err
on the side of over-protection in v1 and let backtesting (Step 12) tune
parameters down if appropriate.

References:
- Almgren & Chriss (2000) — risk-aversion-parameterized execution.
- Federal Reserve "Volcker Rule" — explicit risk constraints alongside
  learned policies. Same architectural pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.market_data.base import NetworkConditions
from app.regime.base import RegimeEstimate
from app.risk.base import MultiHorizonRiskEstimate
from app.routing.base import (
    MultiTokenRoutingDecision,
    RoutingAdjustment,
)


@dataclass(frozen=True)
class RoutingConfig:
    """Tuning knobs for the rule-based router.

    Defaults are AGGRESSIVE — chosen because the cost of under-protection
    in a payment system (tail-loss) is much greater than the cost of
    over-protection (slightly worse slippage in calm markets). Backtesting
    will inform whether to relax.
    """

    # CVaR exclusion threshold: as a fraction of position value.
    cvar_exclusion_pct: float = 0.05
    """If worst-horizon |cvar_dollar| / position_value_usd > this, exclude.
    Default 5% — aggressive. A $1000 payment with $50+ expected tail loss
    is too risky regardless of other factors."""

    # Stablecoin biases by regime.
    calm_stablecoin_bias_bps: float = 0.0
    """No bias in calm markets — let cost optimization run normally."""

    stress_stablecoin_bias_bps: float = 200.0
    """Stress regime: stablecoins get +200 bps preference. Aggressive."""

    shock_stablecoin_bias_bps: float = 500.0
    """Shock regime: stablecoins get +500 bps preference. Aggressive."""

    # Shock-regime non-stable exclusion.
    shock_excludes_non_stables: bool = True
    """In shock regime, drop all non-stablecoins IF at least one stable
    is in the candidate set. Aggressive default: yes."""

    # Congestion-driven liquidity preference.
    congestion_high_threshold: float = 0.6
    """Congestion score (0-1) above which liquidity preference activates."""

    liquidity_deep_usd: float = 5_000_000.0
    """Liquidity depth (USD) above which a token counts as deep."""

    congestion_liquidity_bias_bps: float = 100.0
    """Bias awarded to deep-liquidity tokens during high congestion."""

    def __post_init__(self) -> None:
        if not 0 < self.cvar_exclusion_pct < 1:
            raise ValueError(
                f"cvar_exclusion_pct must be in (0, 1), got {self.cvar_exclusion_pct}"
            )
        for name, val in (
            ("calm_stablecoin_bias_bps", self.calm_stablecoin_bias_bps),
            ("stress_stablecoin_bias_bps", self.stress_stablecoin_bias_bps),
            ("shock_stablecoin_bias_bps", self.shock_stablecoin_bias_bps),
            ("congestion_liquidity_bias_bps", self.congestion_liquidity_bias_bps),
        ):
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {val}")
        if not 0 <= self.congestion_high_threshold <= 1:
            raise ValueError(
                f"congestion_high_threshold must be in [0, 1], got {self.congestion_high_threshold}"
            )
        if self.liquidity_deep_usd <= 0:
            raise ValueError(
                f"liquidity_deep_usd must be positive, got {self.liquidity_deep_usd}"
            )


@dataclass(frozen=True)
class _PerTokenContext:
    """Internal helper bundling everything we need to decide on one token."""

    symbol: str
    is_stable: bool
    risk: MultiHorizonRiskEstimate
    liquidity_depth_usd: float


class RuleBasedRiskAdaptiveRouter:
    """Rule-based implementation of RiskAdaptiveRouter.

    Stateless: every decide() call independently applies the configured
    rules to the supplied inputs. Configuration is fixed at construction.
    """

    def __init__(self, config: RoutingConfig | None = None) -> None:
        self.config = config if config is not None else RoutingConfig()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def decide(
        self,
        regime: RegimeEstimate,
        risk_estimates: dict[str, MultiHorizonRiskEstimate],
        is_stablecoin: dict[str, bool],
        network: NetworkConditions,
        liquidity_depth_usd: dict[str, float] | None = None,
    ) -> MultiTokenRoutingDecision:
        """Apply rules to produce a routing decision for the candidate set.

        Args:
            regime: current market regime classification.
            risk_estimates: CVaR estimates keyed by token symbol.
            is_stablecoin: mapping from symbol to stablecoin flag.
            network: current chain conditions.
            liquidity_depth_usd: per-token liquidity depth. If None, the
                liquidity rule is skipped (assumed deep).

        Returns:
            MultiTokenRoutingDecision with one adjustment per token.
        """
        if not risk_estimates:
            raise ValueError("risk_estimates must contain at least one token")

        # Validate that is_stablecoin covers all tokens.
        missing_stable_flags = [
            s for s in risk_estimates if s not in is_stablecoin
        ]
        if missing_stable_flags:
            raise ValueError(
                f"is_stablecoin missing entries for: {missing_stable_flags}"
            )

        # Build per-token contexts in input order.
        contexts: list[_PerTokenContext] = []
        for symbol, risk in risk_estimates.items():
            depth = (
                liquidity_depth_usd.get(symbol, float("inf"))
                if liquidity_depth_usd is not None
                else float("inf")
            )
            contexts.append(
                _PerTokenContext(
                    symbol=symbol,
                    is_stable=is_stablecoin[symbol],
                    risk=risk,
                    liquidity_depth_usd=depth,
                )
            )

        # Apply rules with progressive relaxation if the result would be empty.
        adjustments = self._apply_rules(
            contexts=contexts,
            regime=regime,
            network=network,
            allow_shock_non_stable_exclusion=self.config.shock_excludes_non_stables,
            allow_cvar_exclusion=True,
        )

        if not any(not a.excluded for a in adjustments):
            # Step 1: drop the shock non-stable exclusion.
            adjustments = self._apply_rules(
                contexts=contexts,
                regime=regime,
                network=network,
                allow_shock_non_stable_exclusion=False,
                allow_cvar_exclusion=True,
            )

        if not any(not a.excluded for a in adjustments):
            # Step 2: drop the CVaR exclusion too. This is the last fallback.
            adjustments = self._apply_rules(
                contexts=contexts,
                regime=regime,
                network=network,
                allow_shock_non_stable_exclusion=False,
                allow_cvar_exclusion=False,
            )

        return MultiTokenRoutingDecision(
            adjustments=tuple(adjustments),
            regime=regime,
        )

    # -------------------------------------------------------------------
    # Rule application
    # -------------------------------------------------------------------

    def _apply_rules(
        self,
        contexts: list[_PerTokenContext],
        regime: RegimeEstimate,
        network: NetworkConditions,
        allow_shock_non_stable_exclusion: bool,
        allow_cvar_exclusion: bool,
    ) -> list[RoutingAdjustment]:
        """Apply the three rule families to each token context.

        The exclusion flags allow the empty-set fallback in decide() to
        progressively relax rules.
        """
        # Determine whether shock-regime non-stable exclusion should fire
        # for this batch (requires at least one stablecoin in the set).
        any_stable = any(c.is_stable for c in contexts)
        shock_exclusion_active = (
            allow_shock_non_stable_exclusion
            and regime.regime == "shock"
            and any_stable
        )

        adjustments: list[RoutingAdjustment] = []
        for ctx in contexts:
            adj = self._decide_one_token(
                ctx=ctx,
                regime=regime,
                network=network,
                shock_exclusion_active=shock_exclusion_active,
                allow_cvar_exclusion=allow_cvar_exclusion,
            )
            adjustments.append(adj)
        return adjustments

    def _decide_one_token(
        self,
        ctx: _PerTokenContext,
        regime: RegimeEstimate,
        network: NetworkConditions,
        shock_exclusion_active: bool,
        allow_cvar_exclusion: bool,
    ) -> RoutingAdjustment:
        """Apply all rules to one token. Exclusions trump biases for output,
        but biases are still recorded for audit when a token is excluded."""

        # --- Exclusion rules ---
        if allow_cvar_exclusion and self._exceeds_cvar_threshold(ctx.risk):
            cvar_pct = abs(ctx.risk.worst_cvar_dollar()) / ctx.risk.position_value_usd
            return self._build_adjustment(
                symbol=ctx.symbol,
                excluded=True,
                exclusion_reason=(
                    f"CVaR breach: worst-case loss "
                    f"{cvar_pct * 100:.2f}% of position exceeds "
                    f"threshold {self.config.cvar_exclusion_pct * 100:.2f}%"
                ),
                bias_bps=0.0,
                bias_reasons=[],
            )

        if shock_exclusion_active and not ctx.is_stable:
            return self._build_adjustment(
                symbol=ctx.symbol,
                excluded=True,
                exclusion_reason=(
                    f"Shock regime: non-stablecoin excluded "
                    f"(stablecoin alternatives available, regime confidence {regime.confidence:.2f})"
                ),
                bias_bps=0.0,
                bias_reasons=[],
            )

        # --- Bias rules (only apply if not excluded) ---
        bias_bps = 0.0
        bias_reasons: list[str] = []

        # Stablecoin preference by regime.
        if ctx.is_stable:
            if regime.regime == "stress":
                bias_bps += self.config.stress_stablecoin_bias_bps
                if self.config.stress_stablecoin_bias_bps > 0:
                    bias_reasons.append(
                        f"Stress regime stablecoin preference: "
                        f"+{self.config.stress_stablecoin_bias_bps:.0f} bps"
                    )
            elif regime.regime == "shock":
                bias_bps += self.config.shock_stablecoin_bias_bps
                if self.config.shock_stablecoin_bias_bps > 0:
                    bias_reasons.append(
                        f"Shock regime stablecoin preference: "
                        f"+{self.config.shock_stablecoin_bias_bps:.0f} bps"
                    )
            # calm: no bias by design.

        # Liquidity preference under congestion.
        if (
            network.congestion_score >= self.config.congestion_high_threshold
            and ctx.liquidity_depth_usd >= self.config.liquidity_deep_usd
        ):
            bias_bps += self.config.congestion_liquidity_bias_bps
            if self.config.congestion_liquidity_bias_bps > 0:
                bias_reasons.append(
                    f"Congestion liquidity preference: "
                    f"+{self.config.congestion_liquidity_bias_bps:.0f} bps "
                    f"(congestion {network.congestion_score:.2f}, "
                    f"liquidity ${ctx.liquidity_depth_usd:,.0f})"
                )

        return self._build_adjustment(
            symbol=ctx.symbol,
            excluded=False,
            exclusion_reason=None,
            bias_bps=bias_bps,
            bias_reasons=bias_reasons,
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _exceeds_cvar_threshold(
        self, risk: MultiHorizonRiskEstimate
    ) -> bool:
        """True if the worst-horizon |CVaR| exceeds threshold pct of position."""
        if risk.position_value_usd <= 0:
            return False
        cvar_dollar = risk.worst_cvar_dollar()
        if cvar_dollar >= 0:
            # Pathological: CVaR is non-negative, which means no downside risk.
            return False
        cvar_pct = abs(cvar_dollar) / risk.position_value_usd
        return cvar_pct > self.config.cvar_exclusion_pct

    @staticmethod
    def _build_adjustment(
        symbol: str,
        excluded: bool,
        exclusion_reason: str | None,
        bias_bps: float,
        bias_reasons: list[str],
    ) -> RoutingAdjustment:
        """Assemble a RoutingAdjustment respecting the contract invariants."""
        # The contract requires:
        # - if bias != 0 then bias_reasons is non-empty
        # - if bias == 0 then bias_reasons must be empty
        # When excluded, biases are recorded for audit but kept consistent.
        if abs(bias_bps) <= 1e-9:
            bias_bps = 0.0
            bias_reasons = []
        return RoutingAdjustment(
            symbol=symbol,
            excluded=excluded,
            exclusion_reason=exclusion_reason,
            score_bias_bps=bias_bps,
            bias_reasons=tuple(bias_reasons),
        )
