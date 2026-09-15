import importlib
import os
from typing import Any, Literal

import upath
import yaml
from pydantic import BaseModel, field_validator, model_validator

from pyxis_portfolio_challenge import PROJECT_ROOT


def fibonacci_cost(n: int, base: float) -> float:
    """
    Total cost for n concurrent readings using Fibonacci multipliers.

    Cumulative sums: n=1→1×, n=2→2×, n=3→4×, n=4→7×, n=5→12× base.
    """
    a, b, total = 1, 1, 0
    for _ in range(n):
        total += a
        a, b = b, a + b
    return total * base


def fibonacci_number(n: int) -> int:
    """
    The n-th Fibonacci number (1-indexed): fib(1)=1, fib(2)=1, fib(3)=2, ...

    Used as the *incremental* multiplier for clinical-site purchases, as opposed
    to ``fibonacci_cost`` which sums the sequence for concurrent readings.
    """
    if n < 1:
        raise ValueError("fibonacci_number is 1-indexed; n must be >= 1")
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return a


class InvestmentLevelParams(BaseModel):
    """Parameters for a single investment level."""

    cost_modifier: float
    speed_modifier: float
    success_modifier: float
    capacity_cost: int
    experience_modifier: float

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class CapacityConfig(BaseModel):
    """Configuration for R&D capacity constraints."""

    enabled: bool
    base_capacity: float
    overage_max_penalty: float  # Max penalty on success rates
    overage_cost_max_penalty: float  # Max penalty on costs
    overage_scaling: Literal["linear", "quadratic"]

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    def calculate_success_modifier(self, capacity_used: float) -> float:
        """
        Calculate the global success modifier based on capacity usage.

        Returns 1.0 if under capacity, decreases linearly/quadratically as
        capacity is exceeded.
        """
        if not self.enabled or capacity_used <= self.base_capacity:
            return 1.0

        overage = capacity_used - self.base_capacity
        overage_ratio = overage / self.base_capacity

        if self.overage_scaling == "quadratic":
            penalty = overage_ratio**2 * self.overage_max_penalty
        else:  # linear
            penalty = overage_ratio * self.overage_max_penalty

        penalty = min(penalty, self.overage_max_penalty)
        return 1.0 - penalty

    def calculate_cost_modifier(self, capacity_used: float) -> float:
        """
        Calculate the global cost modifier based on capacity usage.

        Returns 1.0 if under capacity, increases linearly/quadratically as
        capacity is exceeded (costs go UP when over capacity).
        """
        if not self.enabled or capacity_used <= self.base_capacity:
            return 1.0

        overage = capacity_used - self.base_capacity
        overage_ratio = overage / self.base_capacity

        if self.overage_scaling == "quadratic":
            penalty = overage_ratio**2 * self.overage_cost_max_penalty
        else:  # linear
            penalty = overage_ratio * self.overage_cost_max_penalty

        penalty = min(penalty, self.overage_cost_max_penalty)
        return 1.0 + penalty  # Cost INCREASES when over capacity


class InvestmentLevelsConfig(BaseModel):
    """Configuration for investment levels feature."""

    enabled: bool

    # Investment level definitions
    levels: dict[str, InvestmentLevelParams]

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    def get_level_params(self, level_name: str) -> InvestmentLevelParams:
        """Get parameters for a given investment level name."""
        return self.levels[level_name]


class DropActionConfig(BaseModel):
    """
    Configuration for the standalone drop action.

    When enabled, the investment action space becomes ternary per asset:
      0 = NONE  (don't invest / continue as-is)
      1 = INVEST (start/continue at standard level; only valid for Idle)
      2 = DROP   (remove asset from portfolio entirely; valid for any state)

    Independent of investment_levels — enabling both raises a ValueError.

    Drop fee = round(drop_price_fraction * cost_remaining / drop_price_rounding)
               * drop_price_rounding
    """

    enabled: bool
    # fraction of cost_remaining charged on drop
    drop_price_fraction: float = 0.0
    # round drop fee to nearest N GBP (1 = no rounding)
    drop_price_rounding: int = 1_000_000

    @field_validator("drop_price_fraction")
    @classmethod
    def _fraction_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("drop_price_fraction must be between 0.0 and 1.0")
        return v

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    def calculate_drop_fee(self, cost_remaining: float) -> float:
        """Fee charged for dropping an asset whose trial has this cost_remaining."""
        if self.drop_price_fraction <= 0:
            return 0.0
        raw_fee = self.drop_price_fraction * cost_remaining
        rounding = self.drop_price_rounding
        if rounding > 1:
            return round(raw_fee / rounding) * rounding
        return raw_fee


class ClinicalSitesConfig(BaseModel):
    """
    Configuration for the clinical sites capacity feature.

    Clinical sites are concurrency slots that gate active drug trials: each
    operational site can host one InDevelopment asset at a time (Model B — a
    site is freed whenever its asset leaves InDevelopment, including the one-step
    gap between phases). This hard-caps trial throughput so excess cash cannot
    buy unlimited concurrent trials (the runaway-cash problem).

    Agents grow capacity two ways:
      * ``upgrade`` action: buy one site per step at a Fibonacci-scaled cost,
        usable only after a ``site_development_steps`` build delay.
      * site auction: a PvP first-price sealed-bid auction every
        ``auction_interval_steps`` for an immediately-usable site.

    When an agent requests more trial starts than it has free sites, allocation
    is arbitrated per ``agent_priority`` (see design doc §2): False = ascending
    asset-index order (default); True = the agent emits a per-asset ``priority``
    scalar action and sites go to the highest scores. Ungranted requests are
    costless no-ops; the step is never errored.
    """

    enabled: bool

    # Operational sites each agent starts the episode with.
    starting_sites: int = 3
    # Base cost (GBP) for the Fibonacci purchase curve.
    purchase_base_cost: float = 500_000_000
    # Round purchase cost to nearest N GBP (1 = no rounding).
    purchase_cost_rounding: int = 1_000_000
    # Build delay (steps) before a purchased site becomes operational.
    site_development_steps: int = 2

    # Over-request arbitration: False = pure asset-index order (default);
    # True = agent emits a per-asset priority scalar action.
    agent_priority: bool = False
    # Entropy weight for the priority Gaussian head (only used when
    # agent_priority is True). 1.0 = neutral / stock-SB3 weighting.
    priority_entropy_weight: float = 1.0

    # Site auction (PvP)
    auction_enabled: bool = True
    # A site is auctioned every N steps (= years, since 1 step = 1 year).
    auction_interval_steps: int = 20
    # No site auction before this step (warmup). First auction fires exactly on
    # this step, then every ``auction_interval_steps`` after (e.g. 10, 30, 50).
    auction_min_step: int = 10
    # Continuous bid cap (units matching bd_max_bid).
    site_max_bid: float = 100_000

    @field_validator("starting_sites")
    @classmethod
    def _starting_sites_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("starting_sites must be >= 0")
        return v

    @field_validator("site_development_steps", "auction_interval_steps")
    @classmethod
    def _positive_step_counts(cls, v: int) -> int:
        if v < 1:
            raise ValueError("step-count fields must be >= 1")
        return v

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    def purchase_cost(self, total_owned: int) -> float:
        """
        Fibonacci-scaled cash cost to buy the next site.

        ``total_owned`` is how many sites the agent already owns (operational +
        in-development; auction-won sites count as owned). The k-th purchase
        beyond the starting endowment costs ``purchase_base_cost * fib(k)``,
        i.e. 1×, 1×, 2×, 3×, 5×, 8×, ... for the 1st, 2nd, 3rd, ... purchase
        (the full Fibonacci sequence, including both leading 1s).
        """
        k = max(1, total_owned - self.starting_sites + 1)
        raw = self.purchase_base_cost * fibonacci_number(k)
        rounding = self.purchase_cost_rounding
        if rounding > 1:
            return round(raw / rounding) * rounding
        return raw


class InterimTrialObservationsConfig(BaseModel):
    """
    Configuration for interim trial observations feature.

    This feature enables the agent to observe noisy signals during trial
    execution that become clearer over time, allowing informed early stopping.
    """

    enabled: bool

    # Concentration parameter for Beta distribution (higher = less variance)
    latent_quality_concentration: float

    # Initial noise scale for interim signals (decreases as trial progresses)
    initial_noise_scale: float

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class DistributionalPtrsConfig(BaseModel):
    """
    Configuration for distributional PTRS feature.

    This feature makes PTRS a distribution rather than a point estimate,
    introducing compound uncertainty via correlated TA quality modifiers.
    This creates genuine RL advantage over heuristic-based agents.
    """

    enabled: bool

    # Per-TA variance in quality modifier (hidden state sampled at episode start)
    # Higher variance = more uncertainty about TA quality
    ta_quality_variance: dict[str, float]

    # Per-asset additional noise (independent of TA quality)
    asset_noise_std: float

    # Prior concentration for Beta belief representation
    # Higher = tighter prior (more confident initial estimate)
    prior_concentration: float

    # Observation noise for Bayesian updates (how much trial outcomes vary)
    observation_noise: float

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class TAExperienceConfig(BaseModel):
    """
    Configuration for TA experience system.

    This is decoupled from PTRS uncertainty features and can be enabled
    independently with either uncertain_ptrs, distributional_ptrs, or neither.
    """

    enabled: bool

    # Experience needed to reach full knowledge in one TA
    experience_to_full_knowledge: float

    # Max expertise boost (PTRS bonus for specialists)
    max_expertise_boost: float

    # Experience needed to reach max boost
    experience_to_max_boost: float

    # Multiplicative decay per step (e.g., 0.98 = 2% decay)
    experience_decay_rate: float

    # Hard cap on total experience across all TAs (None = no cap)
    max_total_experience: float | None

    # Phase experience weights (how much experience gained per trial phase completion)
    phase_experience_weights: dict[str, float]

    # Asset arrival bias toward experienced TAs
    asset_arrival_temperature: float

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class UncertainPtrsConfig(BaseModel):
    """
    Configuration for uncertain PTRS feature (point-based with noise).

    Note: This is mutually exclusive with distributional_ptrs.
    Use this for adding Gaussian noise to point estimates.
    """

    enabled: bool

    # TA-specific base noise levels
    ta_noise_config: dict[str, float]

    # Phase noise multipliers (later phases are noisier)
    phase_noise_multipliers: dict[str, float]

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class PtrsReadingsConfig(BaseModel):
    """
    Configuration for per-asset PTRS research readings feature.

    Agents pay Fibonacci-scaled costs to commission
    additional PTRS assessments on individual assets. Each reading returns a
    logit-normal sample centred on the true PTRS, delivered one step later.
    Multiple readings reduce flip probability but cannot analytically recover
    true PTRS. Phase noise scales by distance: next pending trial = 1×σ_base,
    one after = 1.5×, further phases use noise_multipliers[i].
    Mutually exclusive with uncertain_ptrs, distributional_ptrs, ta_experience.
    """

    enabled: bool
    # Fraction of asset.trial.cost_remaining charged per reading (base)
    cost_fraction: float
    # Round base to nearest N GBP before Fibonacci (1 = no rounding)
    cost_rounding: int = 1
    # Upper bound for MultiDiscrete; cash gates the real limit
    action_space_max_readings: int = 10
    sigma_logit_base: float
    # Episode-level noise; defaults to sigma_logit_base if None
    sigma_ep: float | None = None
    noise_multipliers: list[float] = [1.0, 1.5, 2.0]
    # Normalisation cap for sample count in observation
    max_sample_obs: int = 20

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class MarketingConfig(BaseModel):
    """
    Configuration for the marketing spend feature.

    Two complementary mechanics per step:
      - Demand creation: agents top up a shared indication-level pool multiplier.
        Revenue for all drugs in the indication is scaled by the multiplier.
        No hard cap — asymptotes naturally via decay. Slow decay each step.
      - Brand equity: agents build a per-drug quality score that boosts their
        drug's quality in the market share calculation. Faster decay; applies
        to any drug regardless of trial phase.

    Both actions are binary per step (0=don't spend, 1=spend fixed cost).
    Cost = fraction of the drug's max_revenue.
    Marketing is a no-op when disable_market_share_competition is true.
    """

    enabled: bool

    # Demand creation parameters
    dc_cost_fraction: float  # cost per top-up as fraction of drug max_revenue
    dc_step_boost: float  # fixed additive increment to demand_multiplier per spend
    dc_decay_rate: float  # per-step multiplicative decay of headroom toward 1.0

    # Brand equity parameters
    be_cost_fraction: float  # cost per step as fraction of drug max_revenue
    be_boost: float  # brand_score increment per spend action
    be_decay_rate: float  # per-step multiplicative decay of brand_score
    be_effectiveness: float  # quality multiplier per unit of (brand_score − floor)

    @field_validator("dc_step_boost", "dc_decay_rate")
    @classmethod
    def _dc_positive(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("dc_step_boost and dc_decay_rate must be positive")
        return v

    @field_validator("be_decay_rate")
    @classmethod
    def _be_decay_in_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("be_decay_rate must be in (0, 1)")
        return v

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class ApprovalPhaseConfig(BaseModel):
    """Configuration for the regulatory approval phase after Phase 3."""

    enabled: bool
    duration_min: int  # Minimum approval duration in steps
    duration_max: int  # Maximum approval duration in steps
    success_rate_min: float  # Minimum PTRS for approval
    success_rate_max: float  # Maximum PTRS for approval
    cost: float  # Filing fees

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class PricingConfig(BaseModel):
    """
    Configuration for per-drug pricing action.

    When enabled, agents can set price levels for on-market drugs.
    Higher prices increase per-unit revenue but reduce market share
    via demand elasticity. Lower prices capture more share.
    """

    enabled: bool
    levels: list[
        float
    ]  # Revenue multipliers per price level (e.g. [0.60, 0.75, 1.00, 1.20, 1.40, 1.60])
    default_level: (
        int  # Index into levels for default/masked pricing (e.g. 2 = Standard 1.0x)
    )
    elasticity: float  # Demand elasticity: share ~ 1/price^elasticity

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"


class MultiAgentConfig(BaseModel):
    """Multi-agent specific configuration parameters."""

    enabled: bool
    num_agents: int
    # BD market parameters
    bd_enabled: bool
    bd_assets_dir: upath.UPath
    bd_eval_assets_dir: upath.UPath
    bd_base_lambda: float  # Base Poisson λ for BD appearance
    bd_leak_lambda_boost: float  # Added λ per recent leak
    bd_min_step: int  # No BD before this step
    # Continuous bid: per-slot cash bid in GBP millions. <= 0 = pass; highest
    # bid wins and pays its own bid. bd_max_bid is the action-space upper bound;
    # real affordability is gated by cash (an overbid can bankrupt the winner).
    bd_max_bid: float  # Action-space cap for a BD bid, in GBP millions
    bd_max_slots: int  # Max BD assets per step (start with 1)
    # Steps an unwon BD asset stays on market (1 = single-step behaviour)
    bd_persist_steps: int = 1
    bd_phase_weights: list[float]  # Phase 1/2/3 sampling weights
    bd_indication_activity_bias: float  # Weight of activity vs uniform
    # Competition parameters
    exclusivity_period: int
    first_mover_bonus: float
    disable_market_share_competition: bool
    # Intelligence parameters
    alert_history_length: int
    alerts_per_agent: int
    # Event-driven leak probabilities: [Phase 1→2, Phase 2→3, Phase 3→Approval]
    leak_phase_probabilities: list[float]
    # Marketing-spend leak probabilities (opponent sees the spend, not the amount):
    # be = brand equity per drug's indication; dc = demand creation per indication
    be_leak_probability: float = 0.8
    dc_leak_probability: float = 0.8
    # Minimum agent count for demand-creation leaks to activate. DC boosts a
    # public per-indication demand multiplier that is already in the observation,
    # so with few agents an opponent can attribute a rise to the only other
    # spender for free and the leak is redundant. It self-activates once
    # attribution stops being trivial (num_agents >= this threshold).
    dc_leak_min_agents: int = 3
    # Multi-agent reward type: "absolute", "relative_rank", "zero_sum"
    reward_type: str
    reward_scale: float
    # Indication-based market segmentation
    target_drugs_per_indication: float
    on_market_fraction: float
    max_indications_per_ta: int
    indication_spread: float
    indication_drift_speed: float
    # Market congestion penalty
    congestion_exponent: float  # α in 1/n^α, 0 = disabled
    congestion_ramp_steps: (
        int  # Entry positions to reach full penalty (1 = binary incumbent/challenger)
    )
    # Fraction of full penalty applied to incumbent (0 = protected)
    congestion_incumbent_penalty: float

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    def compute_indications_per_ta(self, equilibrium_num_assets: int) -> int:
        """Compute the number of indications per TA based on game parameters."""
        num_tas = 3
        raw = (
            self.num_agents
            * (equilibrium_num_assets / num_tas)
            * self.on_market_fraction
            / self.target_drugs_per_indication
        )
        return min(max(1, round(raw)), self.max_indications_per_ta)


class Config(BaseModel):
    """Configuration model for the investment game environment."""

    equilibrium_num_assets: int
    max_num_assets: int
    asset_arrival_sensitivity_below: float
    asset_arrival_sensitivity_above: float
    starting_cash: float
    horizon: int
    reinvestment_percentage: float
    trial_cost_multiplier: float
    shuffle_order: bool
    mask_first_order_assets: bool
    mask_negative_enpv_assets: bool
    auto_center_rewards: bool
    auto_center_calibration_steps: int
    flatten_obs: bool
    reward_fn: dict[str, Any]
    training_data_dir: upath.UPath
    evaluation_data_dir: upath.UPath
    evaluation_metrics: list[dict[str, str]]
    num_eval_episodes: int
    eval_initial_seed: int
    pyxie_model_root_path: upath.UPath
    warmup_on_reset_steps: int
    warmup_on_reset_policy: str

    # TA Experience system
    ta_experience: TAExperienceConfig | None

    # Uncertain PTRS feature configuration (mutually exclusive with distributional_ptrs)
    uncertain_ptrs: UncertainPtrsConfig | None

    # Investment levels feature configuration
    investment_levels: InvestmentLevelsConfig | None

    # Standalone drop action (independent of investment_levels)
    drop_action: DropActionConfig | None

    # Clinical sites capacity feature (hard throughput cap; runaway-cash sink)
    clinical_sites: ClinicalSitesConfig | None = None

    # Interim trial observations feature configuration
    interim_trial_observations: InterimTrialObservationsConfig | None

    # Distributional PTRS feature configuration (mutually exclusive with uncertain_ptrs)
    distributional_ptrs: DistributionalPtrsConfig | None

    # Marketing spend feature (demand creation + brand equity)
    marketing: MarketingConfig | None

    # Per-asset PTRS research readings feature
    ptrs_readings: PtrsReadingsConfig | None

    # R&D capacity constraints (standalone, independent of investment levels)
    rd_capacity: CapacityConfig

    # Approval phase configuration
    approval_phase: ApprovalPhaseConfig

    # Per-drug pricing configuration
    pricing: PricingConfig

    # Multi-agent environment configuration
    multi_agent: MultiAgentConfig

    class Config:  # noqa: D106
        frozen = True
        extra = "forbid"

    @field_validator("distributional_ptrs", mode="after")
    @classmethod
    def validate_ptrs_mutual_exclusivity(cls, v, info):
        """Validate distributional_ptrs exclusivity."""
        # Check mutual exclusivity with uncertain_ptrs
        uncertain_ptrs = info.data.get("uncertain_ptrs")
        if (
            v is not None
            and v.enabled
            and uncertain_ptrs is not None
            and uncertain_ptrs.enabled
        ):
            raise ValueError(
                "uncertain_ptrs and distributional_ptrs are mutually exclusive. "
                "Enable only one PTRS uncertainty feature at a time."
            )

        # Check mutual exclusivity with interim_trial_observations
        interim_obs = info.data.get("interim_trial_observations")
        if (
            v is not None
            and v.enabled
            and interim_obs is not None
            and interim_obs.enabled
        ):
            raise ValueError(
                "interim_trial_observations and distributional_ptrs "
                "are mutually exclusive. With distributional PTRS, "
                "the distribution itself represents uncertainty - "
                "no hidden 'true' PTRS for interim signals."
            )
        return v

    @field_validator("ptrs_readings", mode="after")
    @classmethod
    def validate_ptrs_readings_exclusivity(cls, v, info):
        """Validate ptrs_readings is mutually exclusive with other PTRS features."""
        if v is None or not v.enabled:
            return v
        for field, label in [
            ("uncertain_ptrs", "uncertain_ptrs"),
            ("distributional_ptrs", "distributional_ptrs"),
            ("ta_experience", "ta_experience"),
        ]:
            other = info.data.get(field)
            if other is not None and other.enabled:
                raise ValueError(
                    f"ptrs_readings and {label} are mutually exclusive. "
                    "Enable only one PTRS uncertainty feature at a time."
                )
        return v

    @model_validator(mode="after")
    def validate_ta_experience_requires_ptrs_feature(self):
        """Validate ta_experience requires a PTRS feature."""
        ta_exp = self.ta_experience
        if ta_exp is not None and ta_exp.enabled:
            uncertain = self.uncertain_ptrs
            distributional = self.distributional_ptrs
            has_ptrs_feature = (uncertain is not None and uncertain.enabled) or (
                distributional is not None and distributional.enabled
            )
            if not has_ptrs_feature:
                raise ValueError(
                    "ta_experience requires either uncertain_ptrs "
                    "or distributional_ptrs to be enabled. "
                    "The expertise boost and PTRS convergence "
                    "mechanics only function with a PTRS "
                    "uncertainty feature active."
                )
        return self

    @model_validator(mode="after")
    def validate_clinical_sites_vs_rd_capacity(self):
        """Clinical sites and rd_capacity are both throughput caps; not both."""
        sites = self.clinical_sites
        if (
            sites is not None
            and sites.enabled
            and self.rd_capacity is not None
            and self.rd_capacity.enabled
        ):
            raise ValueError(
                "clinical_sites and rd_capacity are mutually exclusive. "
                "Both cap trial throughput; enable at most one."
            )
        return self

    def get_pyxie_model_model_path(self, level: int) -> upath.UPath:
        """Get the model file path for the Pyxie model for the given level."""
        if level == -1:
            # preserving old behavior for level -1 to point to level 2 model
            return self.pyxie_model_root_path / "2" / "best_model.zip"
        return self.pyxie_model_root_path / f"{level}" / "best_model.zip"

    def get_pyxie_model_vecnorm_path(self, level: int) -> upath.UPath:
        """Get the vecnorm file path for the Pyxie model for the given level."""
        if level == -1:
            # preserving old behavior for level -1 to point to level 2 model
            return self.pyxie_model_root_path / "2" / "vecnormalize.pkl"
        return self.pyxie_model_root_path / f"{level}" / "vecnormalize.pkl"

    @model_validator(mode="after")
    def resolve_relative_paths(self):
        """Resolve relative asset paths against PROJECT_ROOT."""
        for field in ("training_data_dir", "evaluation_data_dir"):
            path = getattr(self, field)
            if not path.is_absolute():
                object.__setattr__(self, field, PROJECT_ROOT / path)
        ma = self.multi_agent
        if not ma.bd_assets_dir.is_absolute():
            object.__setattr__(ma, "bd_assets_dir", PROJECT_ROOT / ma.bd_assets_dir)
        if not ma.bd_eval_assets_dir.is_absolute():
            object.__setattr__(
                ma, "bd_eval_assets_dir", PROJECT_ROOT / ma.bd_eval_assets_dir
            )
        return self

    @field_validator("reinvestment_percentage", mode="after")
    def validate_reinvestment_percentage(cls, v):
        """Validate that reinvestment_percentage is between 0.0 and 1.0."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("reinvestment_percentage must be between 0.0 and 1.0")
        return v


def from_yaml(
    path: str = f"{PROJECT_ROOT}/pyxis_portfolio_challenge/config.yaml",
) -> Config:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    Config
        The loaded configuration as a Config object.

    """
    with open(os.path.abspath(path), "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def instantiate_from_config(config: Any):
    """
    Instantiate an object from a configuration dictionary.

    This function is now recursive to handle nested objects.

    example generic:
        The following could be defined in the yaml config file:
        ```yaml
        _target_: module.ClassName
        : arg1value
        kwarg1: kwarg1value
        kwarg2: kwarg2value
        ```

        This would come in as the dictionary:
        {"_target_": "module.ClassName", "": arg1value, "kwarg1": kwarg1value, "kwarg2": kwarg2value}

        This function would then import module.ClassName and instantiate it as:
        module.ClassName(arg1value, kwarg1=kwarg1value, kwarg2=kwarg2value)

    specific use case:
        In our use case, this is used to instantiate reward functions for the investment game environment.

        yaml config example:
        ```yaml
        reward_fn:
            _target_: pyxis_portfolio_challenge.environment.reward_functions.LegacyStaticNPVReward

        ```
    """  # noqa: E501
    if isinstance(config, list):
        return [instantiate_from_config(c) for c in config]
    if not isinstance(config, dict) or "_target_" not in config:
        return config

    target = config["_target_"]
    # Split to module and class name
    module_name, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)

    # Recursively instantiate arguments
    kwargs = {
        k: instantiate_from_config(v)
        for k, v in config.items()
        if k != "_target_" and k != ""
    }
    args = [instantiate_from_config(v) for k, v in config.items() if k == ""]

    return klass(*args, **kwargs)


config = from_yaml()
