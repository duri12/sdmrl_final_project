"""
Reinforcement Learning Baseline Comparison for Energy Trading:
Trains and compares PPO, A2C, SAC, and TD3 algorithms on energy market environment
"""

import gymnasium as gym
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, A2C, SAC, TD3
from electric_market_env import EnergyTradingEnv  # Custom environment
from gymnasium.envs.registration import register

# =============================================
# Environment Configuration
# =============================================
register(
    id="EnergyTradingEnv-v0",
    entry_point="electric_market_env:EnergyTradingEnv"
)

# =============================================
# Training Parameters
# =============================================
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# =============================================
# Model Training Utility
# =============================================
def train_model(algorithm, env, timesteps, checkpoint_interval,
                save_path="checkpoints", init_lr=0.0003):
    """
    Generic training function for RL algorithms
    Args:
        algorithm: RL algorithm class (PPO, A2C, SAC, TD3)
        env: Training environment
        timesteps: Total training timesteps
        checkpoint_interval: Save interval in timesteps
        save_path: Directory for model checkpoints
        init_lr: Initial learning rate
    """
    print(f"\n=== Training {algorithm.__name__} ===")
    model = algorithm("MlpPolicy", env, verbose=1, learning_rate=init_lr)

    # Progressive training with checkpointing
    for i in range(0, timesteps, checkpoint_interval):
        print(f"Training interval: {i:7,} → {i + checkpoint_interval:7,}")
        model.learn(
            total_timesteps=checkpoint_interval,
            reset_num_timesteps=False
        )
        checkpoint_file = f"{save_path}/{algorithm.__name__.lower()}_step_{i + checkpoint_interval}"
        model.save(checkpoint_file)
        print(f"Saved checkpoint: {checkpoint_file}")

    # Save final model
    model.save(f"{algorithm.__name__.lower()}_energy_trading")
    print(f"{algorithm.__name__} training completed!\n")
    return model


# =============================================
# Model Loading (Comment/Uncomment as needed)
# =============================================
# To retrain models, uncomment below:
# ppo_model = train_model(PPO, env, timesteps=10000000, checkpoint_interval=100000, init_lr=0.003)
# a2c_model = train_model(A2C, env, timesteps=5000000, checkpoint_interval=50000, init_lr=0.003)
# sac_model = train_model(SAC, env, timesteps=800000, checkpoint_interval=8000, init_lr=0.003)
# td3_model = train_model(TD3, env, timesteps=800000, checkpoint_interval=8000, init_lr=0.003)

# Load pretrained models
print("\n=== Loading Pretrained Models ===")
ppo_model = PPO.load("ppo_energy_trading")
a2c_model = A2C.load("a2c_energy_trading")
sac_model = SAC.load("sac_energy_trading")
td3_model = TD3.load("td3_energy_trading")
print("All models loaded successfully!")


# =============================================
# Evaluation Utilities
# =============================================
def evaluate_model(model, env, episodes=10):
    """
    Evaluates model performance over multiple episodes
    Returns:
        - Mean reward
        - Reward standard deviation
        - List of individual episode rewards
    """
    print(f"\nEvaluating {model.__class__.__name__}...")
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode {episode + 1:2d}: {episode_reward:8.2f}")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nEvaluation Summary ({model.__class__.__name__}):")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Deviation: {std_reward:.2f}")

    return mean_reward, std_reward, rewards


# =============================================
# Visualization Configuration
# =============================================
def configure_visuals():
    """Sets up consistent visualization parameters"""
    plt.style.use('seaborn')
    sns.set(style="whitegrid", font_scale=1.2)
    return sns.color_palette("husl", 4)


def create_comparison_plot(labels, means, stds, palette):
    """Generates publication-quality comparison plot"""
    plt.figure(figsize=(12, 7), dpi=150)

    # Create enhanced bar plot
    bars = plt.bar(labels, means, yerr=stds, capsize=8,
                   edgecolor='black', linewidth=1.2, alpha=0.9,
                   color=palette, error_kw={'elinewidth': 2, 'capthick': 2})

    # Add value annotations
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() * 1.02,
                 f'{mean:.0f} ± {std:.0f}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color='dimgrey')

    # Style enhancements
    plt.title('Energy Trading Performance Comparison\nReinforcement Learning Algorithms',
              fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=14, labelpad=15, fontweight='semibold')
    plt.ylabel('Average Total Reward ($)', fontsize=14, labelpad=15, fontweight='semibold')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12)

    # Final touches
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(0, max(means) * 1.2)
    plt.legend([plt.Line2D([0], [0], color='gray', lw=2)],
               ['Error bars show 1 standard deviation'], loc='upper right')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()


# =============================================
# Main Execution
# =============================================
if __name__ == "__main__":
    # Initialize environment
    env = gym.make("EnergyTradingEnv-v0")

    # Run evaluations
    ppo_perf, ppo_std, ppo_rewards = evaluate_model(ppo_model, env, episodes=24)
    a2c_perf, a2c_std, a2c_rewards = evaluate_model(a2c_model, env, episodes=24)
    sac_perf, sac_std, sac_rewards = evaluate_model(sac_model, env, episodes=24)
    td3_perf, td3_std, td3_rewards = evaluate_model(td3_model, env, episodes=24)

    # Prepare visualization data
    labels = ["PPO", "A2C", "SAC", "TD3"]
    means = [ppo_perf, a2c_perf, sac_perf, td3_perf]
    stds = [ppo_std, a2c_std, sac_std, td3_std]
    palette = configure_visuals()

    # Generate and save plots
    create_comparison_plot(labels, means, stds, palette)
    plt.savefig('baseline_algorithm_comparison.png', bbox_inches='tight', dpi=600)
    plt.show()

    # Print final results
    print("\n=== Final Performance Summary ===")
    print(f"PPO:   {ppo_perf:.2f} ± {ppo_std:.2f}")
    print(f"A2C:   {a2c_perf:.2f} ± {a2c_std:.2f}")
    print(f"SAC:   {sac_perf:.2f} ± {sac_std:.2f}")
    print(f"TD3:   {td3_perf:.2f} ± {td3_std:.2f}")