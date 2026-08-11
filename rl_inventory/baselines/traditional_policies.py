"""
baselines/traditional_policies.py
===================================
Traditional Inventory Management Baselines for comparison with DQN.

Implements 3 classical policies (mandatory for thesis benchmark):
  1. EOQ  -- Economic Order Quantity (Harris, 1913)
  2. (s,S) Policy -- Reorder-point / Order-up-to policy
  3. Newsvendor Model -- Single-period critical fractile solution

All policies expose the same interface:
    policy.get_action(env_state) -> np.ndarray (n_pairs,) of order indices

This allows plug-and-play evaluation in the same environment loop
used by the DQN agent, ensuring a fair comparison.

References (thesis citations):
-------------------------------
- EOQ: Harris, F.W. (1913). How many parts to make at once.
       Factory, The Magazine of Management, 10(2), 135-136.
- (s,S): Scarf, H. (1960). The optimality of (s,S) policies in dynamic
         inventory problems. Mathematical Methods in the Social Sciences.
- Newsvendor: Arrow, K.J., Harris, T., Marschak, J. (1951).
              Optimal inventory policy. Econometrica, 19(3), 250-272.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any
from scipy import stats


# ===========================================================================
# Base class (shared interface)
# ===========================================================================

class BasePolicy:
    """
    Abstract base class for all inventory policies.

    All policies take a demand history summary and return an action
    array compatible with the MultiWarehouseInventoryEnv action space.
    """

    def __init__(self, n_pairs: int, n_action_levels: int = 6,
                 order_levels: Optional[list] = None):
        """
        Parameters
        ----------
        n_pairs : int
            Number of warehouse-SKU pairs.
        n_action_levels : int
            Number of discrete order levels (must match env).
        order_levels : list
            Actual order quantities corresponding to action indices.
            Default: [0, 10, 20, 30, 40, 50].
        """
        self.n_pairs = n_pairs
        self.n_action_levels = n_action_levels
        self.order_levels = np.array(
            order_levels if order_levels else [0, 10, 20, 30, 40, 50],
            dtype=np.float32,
        )

    def _qty_to_action(self, order_qty: np.ndarray) -> np.ndarray:
        """
        Convert continuous order quantities to discrete action indices.

        Finds the closest level in self.order_levels for each pair.

        Parameters
        ----------
        order_qty : np.ndarray, shape (n_pairs,)
            Desired order quantities in units.

        Returns
        -------
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        actions = np.zeros(self.n_pairs, dtype=np.int32)
        for i, qty in enumerate(order_qty):
            # Snap to nearest discrete level
            idx = int(np.argmin(np.abs(self.order_levels - qty)))
            actions[i] = idx
        return actions

    def get_action(self, inventory: np.ndarray, demand_hist: np.ndarray,
                   **kwargs) -> np.ndarray:
        """
        Compute order action.

        Parameters
        ----------
        inventory : np.ndarray, shape (n_pairs,)
            Current inventory levels (flattened).
        demand_hist : np.ndarray, shape (n_pairs, lookback)
            Demand history for each pair.

        Returns
        -------
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        raise NotImplementedError

    def reset(self):
        """Reset any internal state at the start of a new episode."""
        pass


# ===========================================================================
# 1. EOQ -- Economic Order Quantity
# ===========================================================================

class EOQPolicy(BasePolicy):
    """
    Economic Order Quantity (EOQ) Policy.

    The EOQ formula (Harris, 1913) minimizes the sum of ordering and holding
    costs for a deterministic, constant-demand setting:

        Q* = sqrt(2 × D × K / h)

    where:
        D = average daily demand (units/day)
        K = fixed ordering cost per order ($)
        h = holding cost per unit per day ($)

    Adapted for discrete-action multi-warehouse setting:
    - Estimate D from demand history (rolling mean)
    - Compute Q* using the EOQ formula
    - Place an order when inventory falls below a reorder point ROP = D × lead_time
    - Snap Q* to nearest discrete level

    Citation (thesis-worthy):
        Harris, F.W. (1913). How many parts to make at once.
        Factory, The Magazine of Management, 10(2), 135-136.

    Parameters
    ----------
    n_pairs : int
    holding_cost : float -- h, cost per unit per day
    ordering_cost : float -- K, fixed cost per order
    lead_time : float -- average lead time in days
    safety_stock_k : float -- safety stock multiplier (sigma × k)
    """

    def __init__(
        self,
        n_pairs: int,
        holding_cost: float = 1.0,
        ordering_cost: float = 50.0,
        lead_time: float = 2.0,
        safety_stock_k: float = 1.65,   # 95% service level
        n_action_levels: int = 6,
        order_levels: Optional[list] = None,
    ):
        super().__init__(n_pairs, n_action_levels, order_levels)
        self.h = holding_cost
        self.K = ordering_cost
        self.lead_time = lead_time
        self.safety_stock_k = safety_stock_k

    def get_action(
        self,
        inventory: np.ndarray,
        demand_hist: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute EOQ-based order quantities.

        Parameters
        ----------
        inventory : np.ndarray, shape (n_pairs,)
        demand_hist : np.ndarray, shape (n_pairs, lookback)

        Returns
        -------
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        order_qty = np.zeros(self.n_pairs, dtype=np.float32)

        for i in range(self.n_pairs):
            hist = demand_hist[i]                # (lookback,)
            D = float(hist.mean()) + 1e-6        # avg daily demand

            # Safety stock: k × sigma × sqrt(lead_time)
            sigma = float(hist.std()) + 1e-6
            safety_stock = self.safety_stock_k * sigma * np.sqrt(self.lead_time)

            # Reorder point
            ROP = D * self.lead_time + safety_stock

            if inventory[i] <= ROP:
                # EOQ formula: Q* = sqrt(2DK/h)
                eoq = np.sqrt(2.0 * D * self.K / (self.h + 1e-6))
                order_qty[i] = eoq
            else:
                order_qty[i] = 0.0

        return self._qty_to_action(order_qty)

    def get_eoq(self, avg_demand: float) -> float:
        """Return the EOQ for a given average demand (for analysis)."""
        return np.sqrt(2.0 * avg_demand * self.K / (self.h + 1e-6))


# ===========================================================================
# 2. (s, S) Policy -- Reorder-Point / Order-Up-To
# ===========================================================================

class SsPolicyOptimized(BasePolicy):
    """
    (s, S) Inventory Policy.

    One of the most important results in inventory theory (Scarf, 1960):
    For periodic-review inventory with fixed ordering cost, the optimal
    policy is of the (s, S) form:
      - If inventory ≤ s (reorder point):  order up to S (order-up-to level)
      - Otherwise: do not order

    Parameters estimated from demand history:
      s = mu_LT + z_alpha × sigma_LT
      S = s + EOQ
    where mu_LT, sigma_LT are the mean/std of demand during lead time,
    and z_alpha is the service-level z-score.

    Citation (thesis-worthy):
        Scarf, H. (1960). The optimality of (s,S) policies in the dynamic
        inventory problem. Mathematical Methods in the Social Sciences, 196-202.

    Parameters
    ----------
    n_pairs : int
    s_params : dict, optional -- pre-set {pair_idx: (s, S)} for each pair
    service_level : float -- target service level (e.g., 0.95)
    lead_time : float -- average lead time in days
    holding_cost : float
    ordering_cost : float
    """

    def __init__(
        self,
        n_pairs: int,
        service_level: float = 0.95,
        lead_time: float = 2.0,
        holding_cost: float = 1.0,
        ordering_cost: float = 50.0,
        n_action_levels: int = 6,
        order_levels: Optional[list] = None,
        s_params: Optional[Dict[int, tuple]] = None,
    ):
        super().__init__(n_pairs, n_action_levels, order_levels)
        self.service_level = service_level
        self.lead_time = lead_time
        self.h = holding_cost
        self.K = ordering_cost
        self.z = float(stats.norm.ppf(service_level))  # z-score for service level
        self.s_params = s_params  # {pair_idx: (s_val, S_val)}

    def _compute_s_S(
        self, avg_demand: float, std_demand: float
    ) -> tuple:
        """
        Compute (s, S) parameters from demand statistics.

        s = reorder point = mu_LT + z × sigma_LT
        S = order-up-to  = s + EOQ
        """
        mu_lt = avg_demand * self.lead_time
        sigma_lt = std_demand * np.sqrt(self.lead_time)
        s = mu_lt + self.z * sigma_lt
        eoq = np.sqrt(2.0 * avg_demand * self.K / (self.h + 1e-6))
        S = s + eoq
        return float(s), float(S)

    def get_action(
        self,
        inventory: np.ndarray,
        demand_hist: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Apply the (s,S) policy.

        Parameters
        ----------
        inventory : np.ndarray, shape (n_pairs,)
        demand_hist : np.ndarray, shape (n_pairs, lookback)

        Returns
        -------
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        order_qty = np.zeros(self.n_pairs, dtype=np.float32)

        for i in range(self.n_pairs):
            hist = demand_hist[i]
            D = float(hist.mean()) + 1e-6
            sigma = float(hist.std()) + 1e-6

            # Use pre-set params or compute from history
            if self.s_params and i in self.s_params:
                s, S = self.s_params[i]
            else:
                s, S = self._compute_s_S(D, sigma)

            cur_inv = float(inventory[i])
            if cur_inv <= s:
                # Order up to S
                order_qty[i] = max(0.0, S - cur_inv)
            else:
                order_qty[i] = 0.0

        return self._qty_to_action(order_qty)


# ===========================================================================
# 3. Newsvendor Model
# ===========================================================================

class NewsvendorPolicy(BasePolicy):
    """
    Newsvendor Policy (critical fractile model).

    The Newsvendor model (Arrow et al., 1951) finds the optimal order quantity
    for a single-period stochastic demand problem by balancing:
      - Overage cost c_o  (cost of excess inventory)
      - Underage cost c_u (cost of stockout / lost sales)

    Optimal order quantity:
        Q* = F⁻¹(c_u / (c_u + c_o))   (critical fractile)

    where F is the demand CDF (assumed Normal, estimated from history).

    This is extended to a multi-period rolling horizon by applying the
    single-period optimal at each step.

    Citation (thesis-worthy):
        Arrow, K.J., Harris, T., & Marschak, J. (1951).
        Optimal inventory policy. Econometrica, 19(3), 250-272.

    Parameters
    ----------
    n_pairs : int
    holding_cost : float -- c_o (overage cost per excess unit)
    stockout_cost : float -- c_u (underage cost per stockout unit)
    lead_time : float -- average lead time for demand aggregation
    """

    def __init__(
        self,
        n_pairs: int,
        holding_cost: float = 1.0,
        stockout_cost: float = 10.0,
        lead_time: float = 2.0,
        n_action_levels: int = 6,
        order_levels: Optional[list] = None,
    ):
        super().__init__(n_pairs, n_action_levels, order_levels)
        self.c_o = holding_cost
        self.c_u = stockout_cost
        self.lead_time = lead_time

        # Critical fractile (service level implied by costs)
        self.critical_ratio = self.c_u / (self.c_u + self.c_o)
        # z-score for the critical ratio
        self.z_star = float(stats.norm.ppf(self.critical_ratio))

    def get_action(
        self,
        inventory: np.ndarray,
        demand_hist: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Compute Newsvendor optimal order quantity.

        Q* = mu_LT + z* × sigma_LT  (order up to this level if below)
        Order quantity = max(0, Q* - current_inventory)

        Parameters
        ----------
        inventory : np.ndarray, shape (n_pairs,)
        demand_hist : np.ndarray, shape (n_pairs, lookback)

        Returns
        -------
        actions : np.ndarray, shape (n_pairs,), dtype int
        """
        order_qty = np.zeros(self.n_pairs, dtype=np.float32)

        for i in range(self.n_pairs):
            hist = demand_hist[i]
            D = float(hist.mean()) + 1e-6
            sigma = float(hist.std()) + 1e-6

            # Optimal stock level (accounting for lead-time demand)
            mu_lt    = D * self.lead_time
            sigma_lt = sigma * np.sqrt(self.lead_time)
            q_star   = mu_lt + self.z_star * sigma_lt

            # Order to bring inventory up to q_star
            cur_inv = float(inventory[i])
            order_qty[i] = max(0.0, q_star - cur_inv)

        return self._qty_to_action(order_qty)

    @property
    def implied_service_level(self) -> float:
        """Service level implied by the cost ratio."""
        return self.critical_ratio


# ===========================================================================
# Policy runner -- extract state from env info dict
# ===========================================================================

def extract_state_for_policy(
    obs: np.ndarray,
    env_config: Dict[str, Any],
) -> tuple:
    """
    Extract inventory and demand history from the flat observation vector.

    The env observation is structured as:
      [inventory (n_pairs) | demand_hist (n_pairs × lookback) | pipeline | dow]

    Parameters
    ----------
    obs : np.ndarray, shape (obs_dim,)
        Raw observation from the environment.
    env_config : dict
        Must contain 'n_warehouses', 'n_skus', 'lookback', 'max_inventory'.

    Returns
    -------
    inventory : np.ndarray, shape (n_pairs,) -- raw units (denormalized)
    demand_hist : np.ndarray, shape (n_pairs, lookback) -- raw units
    """
    n_w = env_config["n_warehouses"]
    n_s = env_config["n_skus"]
    n_pairs = n_w * n_s
    lookback = env_config["lookback"]
    max_inv = env_config["max_inventory"]

    # Denormalize from [0,1] back to units
    inv_norm = obs[:n_pairs]
    inventory = inv_norm * max_inv

    dem_norm = obs[n_pairs: n_pairs + n_pairs * lookback]
    demand_hist = dem_norm.reshape(n_pairs, lookback)
    # Denormalize demand (approximate -- use max order level as scale)
    max_dem = max(env_config.get("order_levels", [50]))
    demand_hist = demand_hist * max_dem

    return inventory, demand_hist


# ===========================================================================
# Quick test
# ===========================================================================

if __name__ == "__main__":
    import numpy as np

    N_PAIRS = 60
    LOOKBACK = 7

    # Dummy state
    inv = np.random.uniform(50, 200, size=N_PAIRS).astype(np.float32)
    dem = np.random.poisson(15, size=(N_PAIRS, LOOKBACK)).astype(np.float32)

    print("Testing EOQ Policy...")
    eoq = EOQPolicy(n_pairs=N_PAIRS)
    actions = eoq.get_action(inv, dem)
    print(f"  Actions shape: {actions.shape}, sample: {actions[:5]}")

    print("\nTesting (s,S) Policy...")
    ss = SsPolicyOptimized(n_pairs=N_PAIRS)
    actions = ss.get_action(inv, dem)
    print(f"  Actions shape: {actions.shape}, sample: {actions[:5]}")

    print("\nTesting Newsvendor Policy...")
    nv = NewsvendorPolicy(n_pairs=N_PAIRS)
    actions = nv.get_action(inv, dem)
    print(f"  Actions shape: {actions.shape}, sample: {actions[:5]}")
    print(f"  Implied service level: {nv.implied_service_level:.4f}")

    print("\nAll baselines OK!")
