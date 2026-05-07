"""
Abstract contracts for risk-adaptive routing rules.

Per the original RAMHD specification: "When volatility rises, SmartPay

automatically re-routes the transaction toward safer tokens such as

stablecoins, prioritizing transaction reliability over raw cost."

This module is the explicit risk-adaptation layer that translates

regime classifications and CVaR estimates into routing decisions —

biases and exclusions — that downstream Pareto filtering (Step 8)

and the bandit selector (Step 9) consume.

Output shape:

- One RoutingAdjustment per candidate token.

- Adjustments contain:

    - excluded: drop from candidate set entirely (hard rule)

    - score_bias_bps: additive bias in basis points (soft preference)

    - exclusion_reason / bias_reason: human-readable audit trail

Bias semantics (basis points):

- 1 bp = 0.01%. 100 bps = 1%. 1000 bps = 10%.

- Positive bias = make this token MORE attractive to the selector.

- Negative bias = make this token LESS attractive.

- The selector adds the bias to whatever underlying score it uses.

Why rules instead of learned behavior:

- Rules are explicit, auditable, immediately effective from day one.

- Learned behavior (the bandit) takes thousands of transactions to converge.

- During convergence, rules act as guardrails — protecting users from

  the algorithm's exploration mistakes.

- Rules are configurable; backtesting (Step 12) tunes the parameters.

References:

- Federal Reserve "Volcker Rule" — explicit risk constraints alongside

  learned trading behavior. Same pattern.

- Almgren & Chriss (2000) — risk-aversion parameter in optimal execution.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from app.market_data.base import NetworkConditions
from app.regime.base import RegimeEstimate
from app.risk.base import MultiHorizonRiskEstimate


@dataclass(frozen=True)
class RoutingAdjustment:
    """Risk-adaptive adjustment for one candidate token.

    Attributes:

        symbol: token symbol

        excluded: if True, this token is removed from the candidate set

                  entirely. exclusion_reason explains why.

        exclusion_reason: human-readable reason for exclusion (None if

                          not excluded).

        score_bias_bps: additive bias in basis points. Positive = prefer.

                        Applied by the downstream selector to the token's

                        composite score. Ignored if excluded=True.

        bias_reasons: human-readable reasons contributing to the bias.

                      Empty list if no reasons (bias = 0).

    Invariants:

        - If excluded=True, exclusion_reason must be non-empty.

        - If excluded=False, exclusion_reason must be None.

        - bias_reasons is always a list (possibly empty).

    """

    symbol: str
    excluded: bool
    exclusion_reason: Optional[str]
    score_bias_bps: float
    bias_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if self.excluded and not self.exclusion_reason:
            raise ValueError(
                "excluded=True requires a non-empty exclusion_reason"
            )
        if not self.excluded and self.exclusion_reason is not None:
            raise ValueError(
                "excluded=False requires exclusion_reason=None"
            )
        # Empty bias_reasons should mean zero bias and vice versa,
        # within float tolerance.
        if not self.bias_reasons and abs(self.score_bias_bps) > 1e-9:
            raise ValueError(
                f"score_bias_bps={self.score_bias_bps} requires "
                f"at least one entry in bias_reasons"
            )
        if self.bias_reasons and abs(self.score_bias_bps) <= 1e-9:
            raise ValueError(
                f"bias_reasons {self.bias_reasons} requires non-zero score_bias_bps"
            )


@dataclass(frozen=True)
class MultiTokenRoutingDecision:
    """Routing decision for the full candidate set.

    Wraps a list of RoutingAdjustment objects with helpers downstream

    layers can use without re-iterating. Ordering preserves input order.

    """

    adjustments: tuple[RoutingAdjustment, ...]
    regime: RegimeEstimate
    """The regime estimate driving these decisions (audit trail)."""

    def __post_init__(self) -> None:
        if not self.adjustments:
            raise ValueError("adjustments must contain at least one entry")
        symbols_seen = set()
        for adj in self.adjustments:
            if adj.symbol in symbols_seen:
                raise ValueError(f"Duplicate symbol in adjustments: {adj.symbol}")
            symbols_seen.add(adj.symbol)

    def for_symbol(self, symbol: str) -> RoutingAdjustment:
        """Look up the adjustment for one symbol. Raises KeyError if absent."""
        for adj in self.adjustments:
            if adj.symbol == symbol:
                return adj
        raise KeyError(
            f"{symbol} not in routing decision. "
            f"Available: {[a.symbol for a in self.adjustments]}"
        )

    def included_symbols(self) -> tuple[str, ...]:
        """Symbols NOT excluded — i.e., the candidate set after rule pruning."""
        return tuple(a.symbol for a in self.adjustments if not a.excluded)

    def excluded_symbols(self) -> tuple[str, ...]:
        """Symbols excluded — useful for logging and explainability."""
        return tuple(a.symbol for a in self.adjustments if a.excluded)


class RiskAdaptiveRouter(Protocol):
    """Protocol every risk-adaptive router must satisfy.

    A router takes per-token risk estimates plus the current regime and

    network state, and decides per-token whether to exclude and/or bias.

    Implementations may use rules (the v1 default), learned policies, or

    hybrid approaches.

    """

    def decide(
        self,
        regime: RegimeEstimate,
        risk_estimates: dict[str, MultiHorizonRiskEstimate],
        is_stablecoin: dict[str, bool],
        network: NetworkConditions,
    ) -> MultiTokenRoutingDecision:
        """Produce routing adjustments for all candidate tokens.

        Args:

            regime: current market regime classification.

            risk_estimates: CVaR estimates keyed by token symbol.

            is_stablecoin: mapping from token symbol to stablecoin flag,

                           used to decide stablecoin-preference biases.

            network: current chain conditions (gas, congestion, slot time).

        Returns:

            MultiTokenRoutingDecision with one adjustment per token in

            risk_estimates. Order preserved from risk_estimates.keys().

        """
        ...
