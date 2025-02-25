import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC
from electric_market_env import EnergyTradingEnv
from gymnasium.envs.registration import register

# Register the custom environment
register(
    id="EnergyTradingEnv-v0",
    entry_point="electric_market_env:EnergyTradingEnv"
)

# Initialize environment
env = gym.make("EnergyTradingEnv-v0")

# Meta-learning training function (PPO with DynaMITE-RL enhancements)
def train_model_meta(algorithm, env, timesteps=100000, checkpoint_interval=20000, save_path="checkpoints",
                     init_lr=0.003):
    print(f"Training {algorithm.__name__} model with meta-learning (DynaMITE-RL) adaptation...")
    # Initialize model with an initial learning rate
    model = algorithm("MlpPolicy", env, verbose=1, learning_rate=init_lr)
    os.makedirs(save_path, exist_ok=True)
    meta_lr = init_lr
    prev_reward = -float('inf')

    for i in range(0, timesteps, checkpoint_interval):
        print(f"\nTraining from step {i} to {i + checkpoint_interval} with learning rate {meta_lr}...")
        model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)

        # Evaluate current performance using a few episodes (3 episodes here for speed)
        eval_reward, _, _ = evaluate_model(model, env, episodes=3)
        print(f"Evaluation mean reward: {eval_reward}")

        # Meta-update: Adjust learning rate based on improvement
        if eval_reward < prev_reward + 1e-3:  # minimal improvement threshold
            meta_lr *= 0.9
            print("Performance did not improve sufficiently, reducing learning rate to", meta_lr)
        else:
            meta_lr *= 1.05
            print("Performance improved, increasing learning rate to", meta_lr)

        # Update the optimizer's learning rate in the model
        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = meta_lr

        prev_reward = eval_reward
        model.save(f"{save_path}/{algorithm.__name__.lower()}_step_{i + checkpoint_interval}")
        print(f"Checkpoint saved at step {i + checkpoint_interval}")

    print(f"{algorithm.__name__} model training with meta-learning completed.")
    return model


# Evaluation function (unchanged)
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
    print(
        f"{model.__class__.__name__} model evaluation completed. Mean Reward: {mean_reward}, Std Reward: {std_reward}")
    return mean_reward, std_reward, rewards


ppo_dynamite_model = train_model_meta(PPO, env,timesteps=500000, checkpoint_interval=100000)  # PPO with meta-learning (DynaMITE-RL) adaptation

ppo_dynamite_model.save("ppo_dynamite_energy_trading")
print("Final model saved.")

# Load models
ppo_model = PPO.load("ppo_energy_trading")
sac_model = SAC.load("sac_energy_trading")
ppo_dynamite_model = PPO.load("ppo_dynamite_energy_trading")
print("Models loaded.")

# Evaluate all models
ppo_perf, ppo_std, ppo_rewards = evaluate_model(ppo_model, env)
sac_perf, sac_std, sac_rewards = evaluate_model(sac_model, env)
ppo_dynamite_perf, ppo_dynamite_std, ppo_dynamite_rewards = evaluate_model(ppo_dynamite_model, env)

# Visualization of results
labels = ["PPO", "SAC", "PPO+DynaMITE"]
means = [ppo_perf, sac_perf, ppo_dynamite_perf]
stds = [ppo_std, sac_std, ppo_dynamite_std]

plt.figure(figsize=(10, 5))
plt.bar(labels, means, yerr=stds, capsize=5, color=['blue', 'red', 'green'])
plt.xlabel("Algorithm")
plt.ylabel("Average Reward")
plt.title("Comparison of RL Algorithms with DynaMITE-RL Meta-Learning")
plt.savefig("rl_algorithm_comparison.png")
plt.show()

# Print results
print("PPO Performance:", (ppo_perf, ppo_std))
print("SAC Performance:", (sac_perf, sac_std))
print("PPO+DynaMITE Performance:", (ppo_dynamite_perf, ppo_dynamite_std))
