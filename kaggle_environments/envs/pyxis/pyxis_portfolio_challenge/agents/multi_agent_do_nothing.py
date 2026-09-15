"""Do-nothing agent for multi-agent competitive environment."""

import numpy as np


class MultiAgentDoNothingAgent:
    """
    Agent that takes no actions — passes on all investments and BD bids.

    Useful as a baseline or placeholder opponent.
    """

    def __init__(self, agent_name: str, *, env=None):
        """
        Initialise multi-agent do-nothing agent.

        Parameters
        ----------
        agent_name : str
            The agent identifier in the multi-agent environment
            (e.g. ``"pharma_0"``).
        env : MultiAgentInvestmentGameEnv | None
            Environment reference. Can be ``None`` if ``set_env`` is
            called before the first ``__call__``.

        """
        self.agent_name = agent_name
        self.env = env

    def set_env(self, env):
        """Set or update the environment reference."""
        self.env = env

    def __call__(self, obs) -> dict:
        """
        Return a do-nothing action.

        Parameters
        ----------
        obs
            Observation from the environment (unused).

        Returns
        -------
        dict
            Action dict with all-zero ``"investments"`` and ``"bd_bids"``.

        """
        return {
            "investments": np.zeros(
                self.env.max_num_assets, dtype=np.int64
            ),
            "bd_bids": np.zeros(self.env.bd_max_slots, dtype=np.float32),
        }
