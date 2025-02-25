import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, TD3
from electric_market_env import EnergyTradingEnv
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id="EnergyTradingEnv-v0",
    entry_point="electric_market_env:EnergyTradingEnv"
)

# Initialize environment
env = gym.make("EnergyTradingEnv-v0")

# Define models
def train_model(algorithm, env, timesteps, checkpoint_interval, save_path="checkpoints", init_lr=0.003):
    print(f"Training {algorithm.__name__} model...")
    model = algorithm("MlpPolicy", env, verbose=1,learning_rate=init_lr)
    os.makedirs(save_path, exist_ok=True)

    for i in range(0, timesteps, checkpoint_interval):
        print(f"Training from step {i} to {i + checkpoint_interval}...")
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
        model.save(f"{save_path}/{algorithm.__name__.lower()}_step_{i + checkpoint_interval}")
        print(f"Checkpoint saved at step {i + checkpoint_interval}")

    print(f"{algorithm.__name__} model training completed.")
    return model

# Train four RL models with checkpoints
ppo_model = train_model(PPO, env, timesteps=500000, checkpoint_interval=100000)
#a2c_model = train_model(A2C, env, timesteps=30000, checkpoint_interval=5000)
#sac_model = train_model(SAC, env, timesteps=40000, checkpoint_interval=8000)
#td3_model = train_model(TD3, env, timesteps=40000, checkpoint_interval=8000)

# Save final models
ppo_model.save("ppo_energy_trading")
#a2c_model.save("a2c_energy_trading")
#sac_model.save("sac_energy_trading")
#td3_model.save("td3_energy_trading")
#print("Final models saved.")

# Load models
ppo_model = PPO.load("ppo_energy_trading")
a2c_model = A2C.load("a2c_energy_trading")
sac_model = SAC.load("sac_energy_trading")
td3_model = TD3.load("td3_energy_trading")
print("Models loaded.")

# Compare performance
def evaluate_model(model, env, episodes=10):
    print(f"Evaluating {model.__class__.__name__} model...")
    rewards = []
    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"{model.__class__.__name__} model evaluation completed. Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    return mean_reward, std_reward, rewards

# Evaluate all models
ppo_perf, ppo_std, ppo_rewards = evaluate_model(ppo_model, env)
a2c_perf, a2c_std, a2c_rewards = evaluate_model(a2c_model, env)
sac_perf, sac_std, sac_rewards = evaluate_model(sac_model, env)
td3_perf, td3_std, td3_rewards = evaluate_model(td3_model, env)

# Visualization
labels = ["PPO", "A2C", "SAC", "TD3"]
means = [ppo_perf, a2c_perf, sac_perf, td3_perf]
stds = [ppo_std, a2c_std, sac_std, td3_std]

plt.figure(figsize=(10, 5))
plt.bar(labels, means, yerr=stds, capsize=5, color=['blue', 'green', 'red', 'purple'])
plt.xlabel("Algorithm")
plt.ylabel("Average Reward")
plt.title("Comparison of RL Algorithms")
plt.savefig("rl_algorithm_comparison.png")
plt.show()

# Print results
print("PPO Performance:", (ppo_perf, ppo_std))
print("A2C Performance:", (a2c_perf, a2c_std))
print("SAC Performance:", (sac_perf, sac_std))
print("TD3 Performance:", (td3_perf, td3_std))