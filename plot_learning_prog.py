import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3 import SAC as StableSAC
import gymnasium as gym
from electric_market_env import EnergyTradingEnv
from gymnasium.envs.registration import register
from stable_baselines3 import SAC
import torch

# Register the custom environment
register(
    id="EnergyTradingEnv-v0",
    entry_point="electric_market_env:EnergyTradingEnv"
)

env = gym.make("EnergyTradingEnv-v0")

# Define model checkpoint details
checkpoint_intervals = {
    "PPO": (10000000, 100000),
    "A2C": (5000000, 50000),
    "SAC": (800000, 8000),
    "TD3": (800000, 8000),
    "SN_SAC": (800000, 8000),
    "ESAC": (800000, 8000),
    "MSAC":(800000, 8000),
}

checkpoint_dir = "checkpoints"
models = {"PPO": PPO, "A2C": A2C, "TD3": TD3,
          "SAC": StableSAC,"SN_SAC": SAC, "ESAC": SAC, "MSAC": SAC}
eval_episodes = 100  # Increase sample size for averaging


# Function to evaluate a model at a given checkpoint
def evaluate_checkpoint(model_path, env, episodes=20):
    model = None
    try:
        if "ppo" in model_path:
            model = PPO.load(model_path)
        elif "a2c" in model_path:
            model = A2C.load(model_path)
        elif "sac" in model_path:
            model = SAC.load(model_path)
        elif "td3" in model_path:
            model = TD3.load(model_path)
        elif "sn_sac" in model_path:
            model = SAC.load(model_path)
        elif "esac" in model_path:
            model = SAC.load(model_path)
        elif "msac" in model_path:
            model = SAC.load(model_path)
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
        if algo != "ESAC":

            checkpoint_path = os.path.join(checkpoint_dir, f"{algo.lower()}_step_{step}.zip")
            if os.path.exists(checkpoint_path):
                avg_reward = evaluate_checkpoint(checkpoint_path, env, eval_episodes)
                if avg_reward is not None:
                    normalized_step = (step / timesteps) * 100  # Normalize step (0-100%)
                    learning_progress[algo].append((normalized_step, avg_reward))
            else:
                print(checkpoint_path)
        else:
            checkpoint_paths = [os.path.join(checkpoint_dir, f"{algo.lower()}_{i}_step_{step}.zip") for i in range(3)]
            if os.path.exists(checkpoint_paths[0]):
                sum = 0
                for checkpoint_path in checkpoint_paths:
                    avg_reward = evaluate_checkpoint(checkpoint_path, env, eval_episodes)
                    sum += avg_reward
                if sum != 0:
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
