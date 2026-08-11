"""
env/inventory_env.py
====================
Custom Gymnasium environment for Multi-Warehouse Inventory Management.

Design Choices (documented for thesis):
-----------------------------------------
1. ACTION SPACE — Discrete (Factorized):
   Each warehouse-SKU pair independently selects from N_ORDER_LEVELS discrete
   order quantities (e.g., {0, 10, 20, 30, 40, 50} units). This avoids the
   intractable joint action space (6^60 for 2 warehouses × 30 SKUs) by using
   factorized/independent action selection — a standard technique in multi-agent
   RL literature (Sunehag et al., 2018 — VDN; Rashid et al., 2018 — QMIX).
   In our single-agent formulation, the DQN outputs Q-values for each pair
   independently and selects argmax per pair.

2. STATE SPACE:
   - Current inventory levels (n_warehouses × n_skus)
   - Recent demand history (lookback=7 days) for each warehouse-SKU pair
   - Pipeline inventory: units already ordered but not yet arrived (lead time)
   - Day-of-week encoding (7-dim one-hot) for seasonality

3. REWARD FUNCTION:
   R_t = -(h * sum(inventory_t) + p * sum(stockout_t) + K * sum(I(order_t > 0)))
   where:
     h = holding cost per unit per day
     p = stockout (penalty) cost per unit of unmet demand
     K = fixed ordering cost per order placed
   Additionally penalized if inventory exceeds max_capacity.

4. TRANSITION DYNAMICS:
   - Demand is realized from real M5 data (or synthetic if not available)
   - Orders placed at time t arrive at t + lead_time (stochastic lead time)
   - Inventory is capped at max_capacity (warehouse constraint)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Default configuration — override via EnvConfig dict
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Inventory dimensions
    "n_warehouses": 2,
    "n_skus": 30,

    # Cost parameters
    "holding_cost": 1.0,        # h: cost per unit held per timestep
    "stockout_cost": 10.0,      # p: cost per unit of unmet demand
    "ordering_cost": 50.0,      # K: fixed cost per order placed (per sku)
    "overflow_penalty": 5.0,    # penalty per unit over max capacity

    # Warehouse constraints
    "max_inventory": 500,       # max units per warehouse-SKU slot
    "initial_inventory": 100,   # starting inventory level

    # Order quantity levels (discrete action mapping)
    # Agent chooses index 0..5 → mapped to units below
    "order_levels": [0, 10, 20, 30, 40, 50],

    # Lead time (days): stochastic U[min, max]
    "lead_time_min": 1,
    "lead_time_max": 3,

    # Demand history lookback window (days)
    "lookback": 7,

    # Episode length (days)
    "episode_length": 112,      # 16 weeks × 7 days

    # Random seed
    "seed": 42,
}


class MultiWarehouseInventoryEnv(gym.Env):
    """
    Multi-Warehouse Inventory Management Environment.

    Simulates inventory decisions for (n_warehouses × n_skus) warehouse-SKU
    pairs over a finite-horizon episode. Demand is driven by real M5 Walmart
    data or synthetic data if M5 is unavailable.

    Attributes
    ----------
    config : dict
        Environment configuration (costs, dimensions, lead time, etc.)
    demand_data : np.ndarray, shape (T, n_warehouses, n_skus)
        Pre-loaded daily demand series. T must be >= episode_length + lookback.
    n_pairs : int
        Total number of warehouse-SKU pairs = n_warehouses × n_skus.
    n_action_levels : int
        Number of discrete order quantity choices per pair.
    observation_space : gym.Space
        Flat Box observation space.
    action_space : gym.Space
        MultiDiscrete — one discrete choice per warehouse-SKU pair.

    Example
    -------
    >>> import numpy as np
    >>> from env.inventory_env import MultiWarehouseInventoryEnv
    >>> env = MultiWarehouseInventoryEnv()  # uses synthetic demand
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        demand_data: Optional[np.ndarray] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the environment.

        Parameters
        ----------
        config : dict, optional
            Override any key in DEFAULT_CONFIG.
        demand_data : np.ndarray, optional
            Shape (T, n_warehouses, n_skus). If None, uses synthetic demand
            (Poisson with random means) for testing/debugging.
        render_mode : str, optional
            'human' for console print, 'ansi' for string return.
        """
        super().__init__()

        # Merge user config with defaults
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.render_mode = render_mode

        # Convenience aliases
        self.n_w = self.config["n_warehouses"]
        self.n_s = self.config["n_skus"]
        self.n_pairs = self.n_w * self.n_s
        self.lookback = self.config["lookback"]
        self.episode_len = self.config["episode_length"]
        self.order_levels = np.array(self.config["order_levels"], dtype=np.float32)
        self.n_action_levels = len(self.order_levels)
        self.lead_min = self.config["lead_time_min"]
        self.lead_max = self.config["lead_time_max"]
        self.max_inv = self.config["max_inventory"]
        self.init_inv = self.config["initial_inventory"]

        # Set up demand data
        self._setup_demand(demand_data)

        # -----------------------------------------------------------------
        # Action Space: MultiDiscrete
        # Each of the n_pairs warehouse-SKU pairs independently chooses
        # an index in [0, n_action_levels - 1].
        # MultiDiscrete([n_action_levels] * n_pairs)
        # -----------------------------------------------------------------
        self.action_space = spaces.MultiDiscrete(
            [self.n_action_levels] * self.n_pairs
        )

        # -----------------------------------------------------------------
        # Observation Space: flat Box
        # Components:
        #   [0 : n_pairs]                    — current inventory (normalized)
        #   [n_pairs : n_pairs*(lookback+1)] — demand history (normalized)
        #   [... : ... + n_pairs*lead_max]   — pipeline inventory (normalized)
        #   [... : ... + 7]                  — day-of-week one-hot
        # -----------------------------------------------------------------
        self.inv_dim = self.n_pairs                           # current stock
        self.dem_dim = self.n_pairs * self.lookback           # demand history
        self.pip_dim = self.n_pairs * self.lead_max           # pipeline
        self.dow_dim = 7                                      # day of week
        self.obs_dim = self.inv_dim + self.dem_dim + self.pip_dim + self.dow_dim

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Normalization constant
        self._norm_inv = float(self.max_inv)
        self._norm_dem = float(self.order_levels.max())

        # Internal state (initialized in reset)
        self.inventory: np.ndarray = None       # (n_w, n_s)
        self.pipeline: np.ndarray = None        # (lead_max, n_w, n_s) — in-transit
        self.demand_history: np.ndarray = None  # (lookback, n_w, n_s)
        self.t: int = 0                         # current timestep
        self.start_idx: int = 0                 # episode start index in demand_data
        self.episode_cost: float = 0.0

        # Tracking metrics for render/evaluate
        self._episode_stockouts: float = 0.0
        self._episode_holding: float = 0.0
        self._episode_ordering: float = 0.0

        # RNG
        self.np_random = np.random.default_rng(self.config["seed"])

    # ------------------------------------------------------------------
    # Demand Setup
    # ------------------------------------------------------------------

    def _setup_demand(self, demand_data: Optional[np.ndarray]) -> None:
        """
        Load or generate demand data.

        Parameters
        ----------
        demand_data : np.ndarray or None
            If provided, shape must be (T, n_warehouses, n_skus).
            If None, generates synthetic Poisson demand.
        """
        required_len = self.episode_len + self.lookback + self.lead_max + 10

        if demand_data is not None:
            assert demand_data.ndim == 3, "demand_data must be 3D: (T, n_w, n_s)"
            assert demand_data.shape[1] == self.n_w, \
                f"demand_data.shape[1]={demand_data.shape[1]} != n_warehouses={self.n_w}"
            assert demand_data.shape[2] == self.n_s, \
                f"demand_data.shape[2]={demand_data.shape[2]} != n_skus={self.n_s}"
            assert demand_data.shape[0] >= required_len, \
                f"Need at least {required_len} time steps, got {demand_data.shape[0]}"
            self.demand_data = demand_data.astype(np.float32)
            self.T = demand_data.shape[0]
        else:
            # Synthetic Poisson demand -- for testing without M5 data
            print("[MultiWarehouseInventoryEnv] No demand_data provided."
                  " Using synthetic Poisson demand (mean ~ U[5, 30]) for debugging.")
            self.T = required_len + 200
            means = self.np_random.uniform(5, 30, size=(self.n_w, self.n_s))
            self.demand_data = self.np_random.poisson(
                means, size=(self.T, self.n_w, self.n_s)
            ).astype(np.float32)

        # Maximum demand for normalization
        self._norm_dem = max(float(self.demand_data.max()), 1.0)

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to the start of a new episode.

        Parameters
        ----------
        seed : int, optional
            Overrides the config seed for reproducibility.
        options : dict, optional
            'start_idx': int — force episode start index in demand_data.

        Returns
        -------
        observation : np.ndarray, shape (obs_dim,)
        info : dict
        """
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Choose a random starting index in the demand time series
        max_start = self.T - self.episode_len - self.lead_max - 10
        if options and "start_idx" in options:
            self.start_idx = int(options["start_idx"])
        else:
            self.start_idx = int(self.np_random.integers(self.lookback, max_start))

        self.t = 0

        # Initialize inventory to a value around init_inv with small noise
        self.inventory = np.full(
            (self.n_w, self.n_s),
            fill_value=self.init_inv,
            dtype=np.float32,
        )
        noise = self.np_random.uniform(-10, 10, size=(self.n_w, self.n_s))
        self.inventory = np.clip(self.inventory + noise, 0, self.max_inv).astype(np.float32)

        # Pipeline: units in-transit, indexed by [days_until_arrival, w, s]
        self.pipeline = np.zeros((self.lead_max, self.n_w, self.n_s), dtype=np.float32)

        # Initialize demand history from data (lookback days before start_idx)
        hist_start = self.start_idx - self.lookback
        if hist_start < 0:
            pad = np.zeros((-hist_start, self.n_w, self.n_s), dtype=np.float32)
            real = self.demand_data[0:self.start_idx]
            self.demand_history = np.concatenate([pad, real], axis=0)
        else:
            self.demand_history = self.demand_data[hist_start:self.start_idx].copy()

        # Reset episode metrics
        self.episode_cost = 0.0
        self._episode_stockouts = 0.0
        self._episode_holding = 0.0
        self._episode_ordering = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one timestep (one day) of the environment.

        Parameters
        ----------
        action : np.ndarray, shape (n_pairs,), dtype int
            Each element is an index into self.order_levels.
            E.g., action[i] = 2 → order order_levels[2] = 20 units for pair i.

        Returns
        -------
        observation : np.ndarray
        reward : float
        terminated : bool (always False — we use truncated for horizon)
        truncated : bool (True when episode_length is reached)
        info : dict
        """
        assert self.inventory is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=np.int32)
        assert action.shape == (self.n_pairs,), \
            f"Expected action shape ({self.n_pairs},), got {action.shape}"

        # ---- 1. Receive pipeline arrivals (lead_time step) ---------------
        # Orders placed with lead_time=1 arrive today (pipeline index 0)
        arrivals = self.pipeline[0].copy()          # (n_w, n_s)
        self.inventory = np.minimum(
            self.inventory + arrivals, self.max_inv
        )

        # Shift pipeline: day 1 becomes day 0, etc.
        self.pipeline = np.roll(self.pipeline, shift=-1, axis=0)
        self.pipeline[-1] = 0.0                     # clear last slot

        # ---- 2. Place new orders -----------------------------------------
        order_units = self.order_levels[action]     # (n_pairs,) float
        order_matrix = order_units.reshape(self.n_w, self.n_s)  # (n_w, n_s)

        # Sample stochastic lead times per warehouse-SKU pair
        lead_times = self.np_random.integers(
            self.lead_min, self.lead_max + 1, size=(self.n_w, self.n_s)
        )  # (n_w, n_s), each in [lead_min, lead_max]

        # Add orders to pipeline at appropriate lead-time slot
        for w in range(self.n_w):
            for s in range(self.n_s):
                lt = lead_times[w, s] - 1           # 0-indexed
                self.pipeline[lt, w, s] += order_matrix[w, s]

        # ---- 3. Realize demand -------------------------------------------
        data_idx = self.start_idx + self.t
        demand = self.demand_data[data_idx].copy()  # (n_w, n_s)

        # Fulfill demand
        fulfilled = np.minimum(self.inventory, demand)
        stockout = demand - fulfilled               # unmet demand (n_w, n_s)
        self.inventory = self.inventory - fulfilled

        # Update demand history (FIFO rolling window)
        self.demand_history = np.concatenate(
            [self.demand_history[1:], demand[np.newaxis, :, :]], axis=0
        )  # (lookback, n_w, n_s)

        # ---- 4. Compute reward (negative total cost) --------------------
        holding_cost = (
            self.config["holding_cost"] * np.sum(self.inventory)
        )
        stockout_cost = (
            self.config["stockout_cost"] * np.sum(stockout)
        )
        # Fixed ordering cost: charged per order placed (>0 units)
        n_orders_placed = np.sum(order_matrix > 0)
        ordering_cost = self.config["ordering_cost"] * n_orders_placed

        # Overflow penalty: units that would have exceeded max_inv (already capped)
        overflow = np.sum(np.maximum(self.inventory + arrivals - self.max_inv, 0.0))
        overflow_cost = self.config["overflow_penalty"] * overflow

        total_cost = holding_cost + stockout_cost + ordering_cost + overflow_cost
        reward = -total_cost

        # ---- 5. Update episode metrics -----------------------------------
        self.episode_cost += total_cost
        self._episode_stockouts += float(np.sum(stockout))
        self._episode_holding += holding_cost
        self._episode_ordering += ordering_cost

        # ---- 6. Advance timestep ----------------------------------------
        self.t += 1
        terminated = False
        truncated = (self.t >= self.episode_len)

        obs = self._get_obs()
        info = self._get_info(
            demand=demand,
            stockout=stockout,
            order=order_matrix,
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            ordering_cost=ordering_cost,
        )

        if self.render_mode == "human":
            self._render_human(info)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation Builder
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Build and return the normalized observation vector.

        Returns
        -------
        obs : np.ndarray, shape (obs_dim,), dtype float32
            All components normalized to approximately [0, 1].
        """
        # (1) Current inventory — normalized by max_inventory
        inv_flat = (self.inventory.flatten() / self._norm_inv).astype(np.float32)

        # (2) Demand history — shape (lookback, n_w, n_s) → flatten → normalize
        dem_flat = (
            self.demand_history.reshape(-1) / self._norm_dem
        ).astype(np.float32)

        # (3) Pipeline inventory — sum over lead-time dimension per (w, s)
        pipeline_total = self.pipeline.sum(axis=0)          # (n_w, n_s)
        pip_flat = (
            pipeline_total.flatten() / self._norm_inv
        ).astype(np.float32)
        # Pad/trim to pip_dim
        pip_flat_full = np.zeros(self.pip_dim, dtype=np.float32)
        fill_len = min(len(pip_flat), self.pip_dim)
        pip_flat_full[:fill_len] = pip_flat[:fill_len]

        # (4) Day-of-week one-hot
        day_idx = (self.start_idx + self.t) % 7
        dow = np.zeros(7, dtype=np.float32)
        dow[day_idx] = 1.0

        obs = np.concatenate([inv_flat, dem_flat, pip_flat_full, dow])
        assert obs.shape == (self.obs_dim,), \
            f"Obs shape mismatch: {obs.shape} vs {(self.obs_dim,)}"
        return obs

    # ------------------------------------------------------------------
    # Info Dict
    # ------------------------------------------------------------------

    def _get_info(self, **kwargs) -> Dict[str, Any]:
        """Build info dictionary for logging and evaluation."""
        info = {
            "timestep": self.t,
            "episode_cost": self.episode_cost,
            "episode_stockouts": self._episode_stockouts,
            "episode_holding_cost": self._episode_holding,
            "episode_ordering_cost": self._episode_ordering,
            "inventory_mean": float(self.inventory.mean()),
            "inventory_total": float(self.inventory.sum()),
        }
        info.update(kwargs)
        return info

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        """
        Render the current state of the environment.

        Returns
        -------
        str or None depending on render_mode.
        """
        if self.render_mode == "human":
            self._render_human({})
        elif self.render_mode == "ansi":
            return self._render_ansi()

    def _render_human(self, info: Dict) -> None:
        """Print a compact summary to stdout."""
        print(
            f"[t={self.t:3d}] "
            f"inv_mean={self.inventory.mean():.1f}  "
            f"stockout={self._episode_stockouts:.1f}  "
            f"ep_cost={self.episode_cost:.1f}"
        )

    def _render_ansi(self) -> str:
        """Return a string representation of the current state."""
        lines = [
            "=" * 50,
            f"Timestep: {self.t} / {self.episode_len}",
            f"Inventory (w0, SKU 0..4): {self.inventory[0, :5]}",
            f"Episode cost so far: {self.episode_cost:.2f}",
            f"Total stockouts: {self._episode_stockouts:.1f}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def close(self):
        """Clean up resources."""
        pass

    # ------------------------------------------------------------------
    # Helper: compute service level for evaluation
    # ------------------------------------------------------------------

    def get_service_level(self) -> float:
        """
        Compute service level = (total_demand - total_stockout) / total_demand.

        Should be called after an episode completes.

        Returns
        -------
        float in [0, 1]
        """
        if self.t == 0:
            return 1.0
        total_dem = float(self.demand_data[
            self.start_idx:self.start_idx + self.t
        ].sum())
        if total_dem == 0:
            return 1.0
        return max(0.0, 1.0 - self._episode_stockouts / total_dem)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    print("Creating environment...")
    env = MultiWarehouseInventoryEnv(
        config={"n_warehouses": 2, "n_skus": 5, "episode_length": 14}
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Obs dim:           {env.obs_dim}")

    print("\nRunning gymnasium check_env...")
    check_env(env, warn=True)
    print("check_env PASSED!")

    print("\nRunning one episode with random policy...")
    obs, info = env.reset(seed=0)
    total_reward = 0.0
    for _ in range(14):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Episode total reward: {total_reward:.2f}")
    print(f"Service level: {env.get_service_level():.4f}")
    print("DONE.")
