import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, TD3
import gymnasium as gym
from electric_market_env import EnergyTradingEnv
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id="EnergyTradingEnv-v0",
    entry_point="electric_market_env:EnergyTradingEnv"
)

env = gym.make("EnergyTradingEnv-v0")

# Define model checkpoint details
checkpoint_intervals = {
    "PPO": (500000, 100000),
    "A2C": (300000, 50000),
    "SAC": (40000, 8000),
    "TD3": (40000, 8000),
}

checkpoint_dir = "checkpoints"
models = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3}
eval_episodes = 50  # Increase sample size for averaging


# Function to evaluate a model at a given checkpoint
def evaluate_checkpoint(model_path, env, episodes=20):
    model = None
    try:
        model = PPO.load(model_path) if "ppo" in model_path else \
            A2C.load(model_path) if "a2c" in model_path else \
                SAC.load(model_path) if "sac" in model_path else \
                    TD3.load(model_path)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None

    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    return np.mean(rewards)


# Evaluate all checkpoints
learning_progress = {algo: [] for algo in models.keys()}

for algo, (timesteps, interval) in checkpoint_intervals.items():
    print(f"Evaluating {algo} checkpoints...")
    for step in range(interval, timesteps + interval, interval):
        checkpoint_path = os.path.join(checkpoint_dir, f"{algo.lower()}_step_{step}.zip")
        if os.path.exists(checkpoint_path):
            avg_reward = evaluate_checkpoint(checkpoint_path, env, eval_episodes)
            if avg_reward is not None:
                normalized_step = (step / timesteps) * 100  # Normalize step (0-100%)
                learning_progress[algo].append((normalized_step, avg_reward))

# Plot results
plt.figure(figsize=(10, 6))
for algo, data in learning_progress.items():
    if data:
        steps, rewards = zip(*data)
        plt.plot(steps, rewards, marker='o', label=algo)

plt.xlabel("Training Progress (%)")
plt.ylabel("Average Reward")
plt.title("Normalized Learning Progression of RL Models")
plt.legend()
plt.grid()
plt.show()
