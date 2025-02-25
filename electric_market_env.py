import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class EnergyTradingEnv(gym.Env):
    def __init__(self, storage_capacity=100.0, peak_demand=150.0, peak_price=10.0, max_steps=24):
        super(EnergyTradingEnv, self).__init__()

        # Storage and Market Parameters
        self.storage_capacity = storage_capacity
        self.energy_level = storage_capacity / 2  # Initial storage state
        self.peak_demand = peak_demand
        self.peak_price = peak_price
        self.current_time = 0
        self.max_steps = max_steps  # Maximum number of steps per episode
        self.price_log = []

        # Action space: Charging/discharging within storage limits
        self.action_space = Box(
            low=np.array([-self.storage_capacity], dtype=np.float32),
            high=np.array([self.storage_capacity], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

        # Observation space: [Energy Level, Demand, Price]
        self.observation_space = Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.storage_capacity, np.finfo(np.float32).max, np.finfo(np.float32).max]),
            dtype=np.float32,
        )

        self.set_seed()

    def set_seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.set_seed(seed)
        self.energy_level = self.storage_capacity / 2
        self.current_time = 0
        self.price_log = []
        return self._gather_state(), {}

    def step(self, action):
        self.current_time += 1
        # Clip the action so we don't exceed storage limits.
        action = np.clip(
            action.item() if isinstance(action, np.ndarray) else action,
            -self.energy_level,
            self.storage_capacity - self.energy_level
        )

        demand = self._calculate_demand()
        price = self._calculate_price()
        self.price_log.append(price)
        if len(self.price_log) > 100:
            self.price_log.pop(0)  # keep history

        # Determine the energy used and market sale based on action.
        if action < 0:  # discharging the battery so making profit
            energy_used = min(-action, demand)
            market_sale = max(-action - energy_used, 0)
        else:  # charging the battery so no selling
            energy_used = 0
            market_sale = 0

        # Update energy level
        self.energy_level = np.clip(self.energy_level + action, 0, self.storage_capacity)

        # Calculate profit
        profit = (energy_used * 0.8) + (market_sale * price * 0.9) - (max(0, demand - energy_used) * 0.06)

        # Check if the episode should terminate
        terminated = self.current_time >= self.max_steps
        truncated = False  # You can add additional conditions for truncation if needed.

        return self._gather_state(), profit, terminated, truncated, {
            "demand_met": energy_used,
            "energy_sold": market_sale,
            "current_price": price
        }

    def _gather_state(self):
        # Use the current demand and price for the observation.
        return np.array([self.energy_level, self._calculate_demand(), self._calculate_price()], dtype=np.float32)

    def _calculate_demand(self):
        # Example demand function with added noise.
        return 100 * np.exp(-((self.current_time % 24 - 8) ** 2) / (2 * (2 ** 2))) + \
               120 * np.exp(-((self.current_time % 24 - 18) ** 2) / (2 * (3 ** 2))) + np.random.normal(0, 10)

    def _calculate_price(self):
        # Example price function with added noise.
        return 5 + 3 * np.sin(2 * np.pi * (self.current_time % 24) / 24) + np.random.normal(0, 1)

    def render(self):
        print(
            f"Time: {self.current_time}, Energy Level: {self.energy_level:.2f}, "
            f"Demand: {self._calculate_demand():.2f}, Price: {self._calculate_price():.2f}"
        )