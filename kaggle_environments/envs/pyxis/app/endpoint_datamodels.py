import logging
import uuid
from typing import Literal, Optional

from pydantic import BaseModel

from pyxis_portfolio_challenge.game.asset import AssetState, DrugAsset
from pyxis_portfolio_challenge.game.constants import InvestmentLevel
from pyxis_portfolio_challenge.game.game_state import GameState
from pyxis_portfolio_challenge.game.multi_agent_game import MultiAgentGame
from pyxis_portfolio_challenge.game.shared_market_state import (
    Alert,
    AlertType,
    IndicationMarketState,
    indication_key,
)
from pyxis_portfolio_challenge.game.trial import Trial, TrialPhase, TrialState

logger = logging.getLogger(__name__)

# Action types that can be sent from frontend
ActionType = Literal["invest", "stop", "none", "minimal", "standard", "accelerated"]


class StartGameRequest(BaseModel):
    """Request model for starting a new game."""

    num_assets: int
    max_num_assets: int
    horizon: int
    starting_cash: float
    global_seed: int = 42
    level_idx: int = -1  # Use -1 to indicate non-level plays


class TrialResponse(BaseModel):
    """Response model for a trial."""

    cost_remaining: float
    time_remaining: int
    ptrs: float
    # Interim observation data
    interim_result: Optional[Literal["positive", "negative"]]
    has_interim_observation: bool
    # Distributional PTRS fields
    ptrs_expected: Optional[float]
    ptrs_confidence: Optional[float]
    ptrs_range_low: Optional[float]
    ptrs_range_high: Optional[float]


class DrugAssetResponse(BaseModel):
    """Response model for a drug asset."""

    id: uuid.UUID
    name: str
    therapeutic_area: Literal[
        "oncology", "respiratory and immunology", "vaccines and infectious disease"
    ]
    type: Literal["internal", "BD"]
    indication: int
    indication_name: str
    description: str
    max_revenue: float  # M
    time_until_max_revenue: int  # H
    time_until_patent_expiry: int  # T
    trials: dict[TrialPhase, TrialResponse]
    state: AssetState
    pending_trial_phase: str | None
    time_on_market: int
    cost_this_step: float
    cost_to_invest_this_step: float
    revenue_this_step: float
    enpv: float  # Full expected NPV (business value); NOT the competition score
    cash_enpv: float  # Cash-adjusted eNPV: value that flows to cash and scores
    expected_costs: list[float]
    expected_revenues: list[float]
    eroi: float
    current_investment_level: Literal["none", "minimal", "standard", "accelerated"]
    available_actions: list[ActionType]


class InvestmentLevelConfigResponse(BaseModel):
    """Response model for a single investment level configuration."""

    cost_modifier: float
    speed_modifier: float
    success_modifier: float
    capacity_cost: int
    experience_modifier: float


class InvestmentLevelsConfigResponse(BaseModel):
    """Response model for all investment levels configuration."""

    levels: dict[str, InvestmentLevelConfigResponse]
    base_capacity: float
    overage_max_penalty: float
    overage_cost_max_penalty: float


class GameStateResponse(BaseModel):
    """Response model for the game state."""

    id: uuid.UUID
    cash: float
    time: int
    horizon: int
    assets: dict[uuid.UUID, DrugAssetResponse]
    expired_assets: dict[uuid.UUID, DrugAssetResponse]
    realised_costs: list[float]
    realised_revenues: list[float]
    game_ended: bool
    ended_reason: str | None
    capital_over_time: list[float]
    enpv_over_time: list[float]
    eroi_over_time: list[float]
    # TA experience
    ta_experience: dict[str, float]
    experience_to_full_knowledge: float
    max_total_experience: float | None
    # R&D Capacity
    capacity_used: float
    capacity_base: float
    success_modifier: float
    cost_modifier: float
    # Feature flags
    ta_experience_enabled: bool
    investment_levels_enabled: bool
    interim_observations_enabled: bool
    distributional_ptrs_enabled: bool
    # TA quality estimates (distributional PTRS feature)
    ta_quality: dict[str, dict[str, float]]
    # Investment levels configuration (for info popup)
    investment_levels_config: InvestmentLevelsConfigResponse | None


class LevelResponse(BaseModel):
    """Response model for level information."""

    level_idx: int
    user_has_completed: bool
    num_assets: int
    max_num_assets: int
    horizon: int
    starting_cash: float
    global_seed: int


class AgentResponse(BaseModel):
    """Response model for agent information."""

    name: str
    cost: float


class ComparisonDashboardResponse(BaseModel):
    """Response model for comparison dashboard data."""

    game_id: uuid.UUID
    av_enpv: dict[str, float]
    final_enpv: dict[str, float]
    final_eroi: dict[str, float]
    final_capital: dict[str, float]
    realised_eroi: dict[str, float]
    enpv_over_time: dict[str, list[float]]
    eroi_over_time: dict[str, list[float]]


# --- Multi-Agent Request/Response Models ---


class StartMultiAgentGameRequest(BaseModel):
    """Request model for starting a multi-agent game."""

    num_assets: int
    max_num_assets: int
    horizon: int
    starting_cash: float
    global_seed: int = 42
    num_opponents: int  # 1-3
    opponent_agents: list[str]  # e.g. ["knapsack_agent", "random"]


class MultiAgentStepRequest(BaseModel):
    """Request model for stepping a multi-agent game."""

    investment_actions: dict[uuid.UUID, Optional[ActionType]]
    bd_bids: list[float] = []
    # Per-BD-asset cash bids in GBP (0 = pass); length = num BD assets. The
    # highest bid wins and pays its own bid; there is no affordability check,
    # so an overbid can bankrupt the winner.


class BDAssetResponse(BaseModel):
    """Response model for a BD asset available for bidding."""

    asset_id: uuid.UUID
    name: str
    therapeutic_area: str
    indication: int
    indication_name: str
    max_revenue: float
    time_until_max_revenue: int
    time_until_patent_expiry: int
    trial_phase: str
    ptrs: float
    enpv: float
    cash_enpv: float  # cash-adjusted eNPV; a fair-value anchor for a cash bid


class AlertResponse(BaseModel):
    """Response model for a competitive intelligence alert."""

    step: int
    event_type: str
    agent_id: str
    therapeutic_area: str
    indication: int
    indication_name: str
    details: dict


class IndicationMarketResponse(BaseModel):
    """Response model for an indication market."""

    therapeutic_area: str
    indication: int
    indication_name: str
    first_mover_agent: str | None
    incumbent_agent: str | None
    exclusivity_remaining: int
    active_drugs: dict[str, int]  # agent_id -> count
    player_market_share: float


class OpponentSummaryResponse(BaseModel):
    """Response model for an opponent agent summary."""

    agent_name: str
    display_name: str
    agent_type: str
    cash: float
    num_assets: int
    num_on_market: int
    num_in_development: int
    enpv: float
    cumulative_reward: float
    game_ended: bool
    ended_reason: str | None


class MultiAgentGameStateResponse(BaseModel):
    """Response model for multi-agent game state."""

    game_id: uuid.UUID
    player_agent_name: str
    player_state: GameStateResponse
    bd_assets: list[BDAssetResponse]
    bd_enabled: bool
    alerts: list[AlertResponse]
    indication_markets: list[IndicationMarketResponse]
    opponents: list[OpponentSummaryResponse]
    time: int
    horizon: int
    player_cumulative_reward: float
    player_bankrupt: bool
    game_ended: bool  # True only when ALL agents finished or horizon reached
    ended_reason: str | None
    last_bd_acquisitions: dict[str, list[str]]


class OpponentAgentInfo(BaseModel):
    """Response model for available opponent agent types."""

    id: str
    name: str
    description: str


# --- Multi-Agent Converter Functions ---


def bd_asset_to_response(
    asset: DrugAsset,
    indication_name_map: dict[str, str] | None = None,
    reinvestment_percentage: float = 0.10,
) -> BDAssetResponse:
    """Convert a DrugAsset (BD candidate) to its response format."""
    ind_key = indication_key(asset.therapeutic_area, asset.indication)
    ind_name = indication_name_map.get(ind_key, "") if indication_name_map else ""
    return BDAssetResponse(
        asset_id=asset.id,
        name=asset.name,
        therapeutic_area=asset.therapeutic_area,
        indication=asset.indication,
        indication_name=ind_name,
        max_revenue=asset.max_revenue,
        time_until_max_revenue=asset.time_until_max_revenue,
        time_until_patent_expiry=asset.time_until_patent_expiry,
        trial_phase=asset.trial.phase.value if asset.trial else "unknown",
        ptrs=asset.trial.ptrs if asset.trial else 0.0,
        enpv=asset.enpv,
        cash_enpv=asset.cash_enpv(reinvestment_percentage),
    )


def alert_to_response(
    alert: Alert,
    indication_name_map: dict[str, str],
    name_map: dict[str, str] | None = None,
) -> AlertResponse:
    """Convert an Alert to its response format."""
    ind_key = indication_key(alert.therapeutic_area, alert.indication)
    display_agent_id = (
        name_map.get(alert.agent_id, alert.agent_id) if name_map else alert.agent_id
    )
    return AlertResponse(
        step=alert.step,
        event_type=alert.event_type.value,
        agent_id=display_agent_id,
        therapeutic_area=alert.therapeutic_area,
        indication=alert.indication,
        indication_name=indication_name_map.get(ind_key, ""),
        details=alert.details,
    )


def indication_market_to_response(
    market: IndicationMarketState,
    current_time: int,
    player_agent: str,
    name_map: dict[str, str] | None = None,
) -> IndicationMarketResponse:
    """Convert an IndicationMarketState to its response format."""
    active_drugs_count = {
        (name_map.get(agent_id, agent_id) if name_map else agent_id): len(drug_ids)
        for agent_id, drug_ids in market.active_drugs.items()
    }

    # Calculate player market share (simplified: proportion of drugs)
    total_drugs = sum(active_drugs_count.values())
    player_display = (
        name_map.get(player_agent, player_agent) if name_map else player_agent
    )
    player_drugs = active_drugs_count.get(player_display, 0)
    if total_drugs > 0:
        player_share = player_drugs / total_drugs
    else:
        player_share = 0.0

    # If player has first-mover exclusivity, share is 1.0
    if market.first_mover_agent == player_agent and market.is_in_exclusivity(
        current_time
    ):
        player_share = 1.0
    elif (
        market.first_mover_agent is not None
        and market.first_mover_agent != player_agent
        and market.is_in_exclusivity(current_time)
    ):
        player_share = 0.0

    first_mover_display = (
        name_map.get(market.first_mover_agent, market.first_mover_agent)
        if name_map and market.first_mover_agent
        else market.first_mover_agent
    )

    # Incumbent = agent owning the drug at entry_order[0] (first on market)
    incumbent_agent_id = None
    if market.entry_order:
        incumbent_drug_id = market.entry_order[0]
        for agent_id, drug_ids in market.active_drugs.items():
            if incumbent_drug_id in drug_ids:
                incumbent_agent_id = agent_id
                break
    incumbent_display = (
        name_map.get(incumbent_agent_id, incumbent_agent_id)
        if name_map and incumbent_agent_id
        else incumbent_agent_id
    )

    return IndicationMarketResponse(
        therapeutic_area=market.therapeutic_area,
        indication=market.indication,
        indication_name=market.indication_name,
        first_mover_agent=first_mover_display,
        incumbent_agent=incumbent_display,
        exclusivity_remaining=market.exclusivity_remaining(current_time),
        active_drugs=active_drugs_count,
        player_market_share=player_share,
    )


def multi_agent_game_to_response(
    game: MultiAgentGame,
    player_agent: str,
    opponent_types: list[str],
    opponent_display_names: list[str] | None = None,
    cumulative_rewards: dict[str, float] | None = None,
) -> MultiAgentGameStateResponse:
    """Convert a MultiAgentGame to its response format."""
    player_state = game.agent_states[player_agent]
    ind_name_map = game.shared_market.indication_name_map
    player_state_response = game_state_to_response(player_state, ind_name_map)

    # Build name mapping: pharma_X -> display name
    name_map: dict[str, str] = {player_agent: "You"}
    if opponent_display_names:
        for i, display_name in enumerate(opponent_display_names):
            name_map[f"pharma_{i + 1}"] = display_name
    else:
        for i in range(len(opponent_types)):
            name_map[f"pharma_{i + 1}"] = f"pharma_{i + 1}"

    bd_assets_response = [
        bd_asset_to_response(
            asset, ind_name_map, player_state.reinvestment_percentage
        )
        for asset in game.shared_market.current_bd_assets
    ]

    alert_responses = [
        alert_to_response(alert, ind_name_map, name_map)
        for alert in game.shared_market.alerts
    ]

    indication_market_responses = [
        indication_market_to_response(market, game.time, player_agent, name_map)
        for market in game.shared_market.indication_markets.values()
    ]

    # Build opponent summaries
    opponent_responses = []
    for i, agent_type in enumerate(opponent_types):
        agent_name = f"pharma_{i + 1}"
        if agent_name not in game.agent_states:
            continue
        state = game.agent_states[agent_name]
        num_on_market = sum(
            1 for a in state.assets.values() if a.state == AssetState.OnMarket
        )
        num_in_dev = sum(
            1 for a in state.assets.values() if a.state == AssetState.InDevelopment
        )
        opponent_responses.append(
            OpponentSummaryResponse(
                agent_name=name_map.get(agent_name, agent_name),
                display_name=name_map.get(agent_name, agent_name),
                agent_type=agent_type,
                cash=state.cash,
                num_assets=len(state.assets),
                num_on_market=num_on_market,
                num_in_development=num_in_dev,
                enpv=sum(a.enpv for a in state.assets.values()),
                cumulative_reward=(
                    cumulative_rewards.get(agent_name, 0.0)
                    if cumulative_rewards
                    else 0.0
                ),
                game_ended=state.game_ended,
                ended_reason=state.ended_reason,
            )
        )

    # Derive last BD acquisitions from BD_DEAL alerts at current step
    last_bd_names: dict[str, list[str]] = {}
    for alert in game.shared_market.alerts:
        if alert.event_type == AlertType.BD_DEAL and alert.step == game.time - 1:
            display_id = name_map.get(alert.agent_id, alert.agent_id)
            asset_name = alert.details.get(
                "asset_name", str(alert.details.get("asset_id", ""))[:8]
            )
            last_bd_names.setdefault(display_id, []).append(asset_name)

    # Player bankrupt = player's game_ended flag (bankruptcy or horizon)
    player_bankrupt = player_state.game_ended and (
        player_state.ended_reason is None
        or "horizon" not in player_state.ended_reason.lower()
    )

    # Game is over when ALL agents have ended or horizon is reached
    all_ended = all(s.game_ended for s in game.agent_states.values())
    horizon_reached = game.time >= game.horizon
    game_ended = all_ended or horizon_reached

    # Use player's ended_reason if they're bankrupt, otherwise check horizon
    if player_bankrupt:
        ended_reason = player_state.ended_reason
    elif horizon_reached:
        ended_reason = "Game horizon reached"
    elif all_ended:
        ended_reason = "All agents eliminated"
    else:
        ended_reason = None

    return MultiAgentGameStateResponse(
        game_id=player_state.id,
        player_agent_name=name_map.get(player_agent, player_agent),
        player_state=player_state_response,
        bd_assets=bd_assets_response,
        bd_enabled=game.shared_market.bd_enabled,
        alerts=alert_responses,
        indication_markets=indication_market_responses,
        opponents=opponent_responses,
        time=game.time,
        horizon=game.horizon,
        player_cumulative_reward=(
            cumulative_rewards.get(player_agent, 0.0) if cumulative_rewards else 0.0
        ),
        player_bankrupt=player_bankrupt,
        game_ended=game_ended,
        ended_reason=ended_reason,
        last_bd_acquisitions=last_bd_names,
    )


def trial_to_response(trial: Trial) -> dict[TrialPhase, TrialResponse]:
    """Convert the Trial and prev/subsequent trials to response."""
    _trial = trial
    response_dict = {}
    failure_detected = False
    for phase in TrialPhase:
        if failure_detected:
            # A failure was detected in a previous phase,
            # so all subsequent ptrs are 0.
            response_dict[phase.value] = TrialResponse(
                cost_remaining=0.0,
                time_remaining=0,
                ptrs=0.0,
                interim_result=None,
                has_interim_observation=False,
                ptrs_expected=0.0,
                ptrs_confidence=1.0,
                ptrs_range_low=0.0,
                ptrs_range_high=0.0,
            )
            continue

        if _trial and _trial.phase == phase:
            # Check if the current trial has failed
            if _trial.state == TrialState.PHASE_FAILED:
                failure_detected = True

            # Check for interim observation
            interim_result = None
            has_interim = False
            if hasattr(_trial, "_interim_observation_result"):
                interim_obs = _trial._interim_observation_result
                if interim_obs is not None:
                    has_interim = True
                    interim_result = "positive" if interim_obs else "negative"

            response_dict[phase.value] = TrialResponse(
                cost_remaining=_trial.cost_remaining,
                time_remaining=_trial.time_remaining,
                ptrs=_trial.ptrs,
                interim_result=interim_result,
                has_interim_observation=has_interim,
                ptrs_expected=_trial.ptrs_expected,
                ptrs_confidence=_trial.ptrs_confidence,
                ptrs_range_low=_trial.ptrs_range_low,
                ptrs_range_high=_trial.ptrs_range_high,
            )
            _trial = _trial.next_trial_on_success
        else:
            # previous trial must have succeeded, so use success values
            response_dict[phase.value] = TrialResponse(
                cost_remaining=0.0,
                time_remaining=0,
                ptrs=1.0,
                interim_result=None,
                has_interim_observation=False,
                ptrs_expected=1.0,
                ptrs_confidence=1.0,
                ptrs_range_low=1.0,
                ptrs_range_high=1.0,
            )

    return response_dict


def asset_to_response(
    drug_asset: DrugAsset,
    investment_levels_enabled: bool = False,
    indication_name_map: dict[str, str] | None = None,
    *,
    reinvestment_percentage: float,
) -> DrugAssetResponse:
    """Convert the asset to a response format for the frontend."""
    base = drug_asset.model_dump()
    base["id"] = drug_asset.id

    # Add indication name from map
    ind_key = indication_key(drug_asset.therapeutic_area, drug_asset.indication)
    base["indication_name"] = (
        indication_name_map.get(ind_key, "") if indication_name_map else ""
    )

    # Convert trials to response format
    del base["trial"]
    base["trials"] = trial_to_response(drug_asset.trial)
    # handle pending trial phase logic
    if drug_asset.state == AssetState.OnMarket:
        pending_trial_phase = None
    elif drug_asset.trial.state == TrialState.PHASE_FAILED:
        pending_trial_phase = None
    else:
        pending_trial_phase = drug_asset.trial.phase.value

    # Add properties to response
    base["cost_this_step"] = drug_asset.cost_this_step
    # cost_to_invest_this_step is the cost if you invest this step
    # (only valid for Idle assets)
    if drug_asset.state == AssetState.Idle:
        base["cost_to_invest_this_step"] = drug_asset.cost_to_invest_this_step
    else:
        base["cost_to_invest_this_step"] = 0.0
    base["revenue_this_step"] = drug_asset.revenue_this_step
    # Full eNPV is the real-world business value; cash_enpv is the score-aligned
    # value (revenues scaled by reinvestment_percentage, matching the NCF reward).
    base["enpv"] = drug_asset.enpv
    base["cash_enpv"] = drug_asset.cash_enpv(reinvestment_percentage)
    (
        base["expected_costs"],
        base["expected_revenues"],
    ) = drug_asset.expected_costs_and_revenues
    base["eroi"] = drug_asset.eroi
    base["pending_trial_phase"] = pending_trial_phase

    # Add investment level info
    level_map = {
        InvestmentLevel.NONE: "none",
        InvestmentLevel.MINIMAL: "minimal",
        InvestmentLevel.STANDARD: "standard",
        InvestmentLevel.ACCELERATED: "accelerated",
    }
    current_level = drug_asset.current_investment_level
    base["current_investment_level"] = level_map[current_level]

    # Determine available actions based on asset state
    available_actions: list[ActionType] = []
    if drug_asset.state == AssetState.Idle:
        if investment_levels_enabled:
            available_actions = ["none", "minimal", "standard", "accelerated"]
        else:
            available_actions = ["none", "invest"]
    elif drug_asset.state == AssetState.InDevelopment:
        if investment_levels_enabled:
            available_actions = ["minimal", "standard", "accelerated", "stop"]
        else:
            available_actions = ["invest", "stop"]
    # On Market, Failed, Expired have no actions
    base["available_actions"] = available_actions

    return base


def game_state_to_response(
    game_state: GameState,
    indication_name_map: dict[str, str] | None = None,
) -> GameStateResponse:
    """Convert the game state to a response format for the frontend."""
    logger.debug("Dumping game state model for response.")
    base = game_state.model_dump()
    base["id"] = game_state.id
    if "running_enpv" in base:
        del base["running_enpv"]
    if "running_eroi" in base:
        del base["running_eroi"]

    # Check if features are enabled
    investment_levels_enabled = (
        hasattr(game_state, "_investment_levels_config")
        and game_state._investment_levels_config is not None
        and game_state._investment_levels_config.enabled
    )
    interim_observations_enabled = (
        hasattr(game_state, "_interim_trial_observations_config")
        and game_state._interim_trial_observations_config is not None
        and game_state._interim_trial_observations_config.enabled
    )
    distributional_ptrs_enabled = (
        hasattr(game_state, "_distributional_ptrs_config")
        and game_state._distributional_ptrs_config is not None
        and game_state._distributional_ptrs_config.enabled
    )
    ta_experience_enabled = (
        hasattr(game_state, "_ta_experience_config")
        and game_state._ta_experience_config is not None
        and game_state._ta_experience_config.enabled
    )

    logger.debug("Converting assets to response.")
    # Convert assets to response format
    base["assets"] = {
        asset_id: asset_to_response(
            asset,
            investment_levels_enabled,
            indication_name_map,
            reinvestment_percentage=game_state.reinvestment_percentage,
        )
        for asset_id, asset in game_state.assets.items()
    }
    base["expired_assets"] = {
        asset_id: asset_to_response(
            asset,
            investment_levels_enabled,
            indication_name_map,
            reinvestment_percentage=game_state.reinvestment_percentage,
        )
        for asset_id, asset in {
            **game_state.expired_assets,
            **game_state.failed_assets,
        }.items()
    }

    logger.debug("Adding properties to response.")
    # Add properties to response
    base["game_ended"] = game_state.game_ended
    base["capital_over_time"] = game_state.capital_over_time
    base["enpv_over_time"] = game_state.enpv_over_time
    base["eroi_over_time"] = game_state.eroi_over_time

    # Add TA experience
    base["ta_experience"] = dict(game_state.ta_experience)

    # Add TA experience config values
    ta_exp_config = game_state._ta_experience_config
    if ta_exp_config is not None:
        base["experience_to_full_knowledge"] = (
            ta_exp_config.experience_to_full_knowledge
        )
        base["max_total_experience"] = ta_exp_config.max_total_experience
    else:
        base["experience_to_full_knowledge"] = 0.0
        base["max_total_experience"] = None

    # Add R&D capacity info
    base["capacity_used"] = game_state.capacity_used
    base["capacity_base"] = game_state.capacity_base
    base["success_modifier"] = game_state.success_modifier
    base["cost_modifier"] = game_state.cost_modifier

    # Add feature flags
    base["investment_levels_enabled"] = investment_levels_enabled
    base["interim_observations_enabled"] = interim_observations_enabled
    base["distributional_ptrs_enabled"] = distributional_ptrs_enabled
    base["ta_experience_enabled"] = ta_experience_enabled

    # Add TA quality estimates (distributional PTRS feature)
    if distributional_ptrs_enabled:
        base["ta_quality"] = {
            ta: {
                "estimate": game_state.ta_quality_estimates[ta],
                "confidence": game_state.ta_quality_confidences[ta],
            }
            for ta in [
                "oncology",
                "respiratory and immunology",
                "vaccines and infectious disease",
            ]
        }
    else:
        base["ta_quality"] = {}

    # Add investment levels configuration for info popup
    if investment_levels_enabled:
        inv_config = game_state._investment_levels_config
        base["investment_levels_config"] = InvestmentLevelsConfigResponse(
            levels={
                level_name: InvestmentLevelConfigResponse(
                    cost_modifier=level_params.cost_modifier,
                    speed_modifier=level_params.speed_modifier,
                    success_modifier=level_params.success_modifier,
                    capacity_cost=level_params.capacity_cost,
                    experience_modifier=level_params.experience_modifier,
                )
                for level_name, level_params in inv_config.levels.items()
            },
            base_capacity=game_state._rd_capacity_config.base_capacity
            if game_state._rd_capacity_config
            else 0.0,
            overage_max_penalty=game_state._rd_capacity_config.overage_max_penalty
            if game_state._rd_capacity_config
            else 0.0,
            overage_cost_max_penalty=game_state._rd_capacity_config.overage_cost_max_penalty
            if game_state._rd_capacity_config
            else 0.0,
        )
    else:
        base["investment_levels_config"] = None

    return base
