"""
Reinforcement Learning for Electricity Market Trading:
Implements and compares three SAC variants - SN-SAC, ESAC, and MSAC
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
from torch.nn.utils import spectral_norm
from electric_market_env import EnergyTradingEnv  # Custom environment
from gymnasium.envs.registration import register
import os
import seaborn as sns


# =============================================
# Environment Registration
# =============================================
register(
    id="EnergyTradingEnv-v0",
    entry_point="electric_market_env:EnergyTradingEnv"
)

# =============================================
# Global Configurations
# =============================================
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Create directory for model checkpoints

# Training parameters
TIMESTEPS = 800000
CHECKPOINT_INTERVAL = 8000
INIT_LR = 0.003


# =============================================
# 1. Spectral Normalization SAC (SN-SAC)
# =============================================
class SNSACPolicy(ActorCriticPolicy):
    """
    SAC with Spectral Normalization in the Actor network
    Enhances stability by constraining layer weights
    """

    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # Spectral normalized actor network
        self.actor = nn.Sequential(
            spectral_norm(nn.Linear(self.features_dim, 256)),
            nn.ReLU(),
            spectral_norm(nn.Linear(256, 256)),
            nn.ReLU(),
            spectral_norm(nn.Linear(256, action_space.shape[0])),
            nn.Tanh()  # Outputs in [-1, 1] for continuous actions
        )

        # Standard critic networks
        self.critic = nn.Sequential(
            nn.Linear(self.features_dim + action_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.critic_target = nn.Sequential(
            nn.Linear(self.features_dim + action_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, deterministic=False):
        """Forward pass for actor network"""
        return self.actor(obs), None  # Second output is for stochastic actions


# =============================================
# 2. Ensemble SAC (ESAC)
# =============================================
class ESAC:
    """
    Ensemble SAC with multiple independent SAC models
    Reduces variance through model averaging
    """

    def __init__(self, env, n_models=3):
        self.models = [SAC('MlpPolicy', env, verbose=0, learning_rate=INIT_LR)
                       for _ in range(n_models)]

    def train(self, total_timesteps=TIMESTEPS):
        """Train ensemble members with periodic checkpointing"""
        for i, model in enumerate(self.models):
            print(f"\nTraining ESAC ensemble member {i + 1}/{len(self.models)}")
            for step in range(0, total_timesteps, CHECKPOINT_INTERVAL):
                model.learn(
                    total_timesteps=CHECKPOINT_INTERVAL,
                    reset_num_timesteps=False
                )
                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"esac_{i}_step_{step + CHECKPOINT_INTERVAL}.zip"
                )
                model.save(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

    def predict(self, observation):
        """Average predictions from all ensemble members"""
        actions = [model.predict(observation, deterministic=True)[0]
                   for model in self.models]
        return np.mean(actions, axis=0)


# =============================================
# 3. Meta-SAC (MSAC)
# =============================================
class MSAC(SAC):
    """
    Meta-Learning SAC that trains across multiple environment variations
    Enables rapid adaptation to new market conditions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, learning_rate=INIT_LR, **kwargs)

    def meta_train(self, envs, total_timesteps=TIMESTEPS):
        """Cyclical training across different environment variations"""
        print("\nStarting MSAC meta-training...")
        cumulative_steps = 0
        steps_per_env = total_timesteps // len(envs)

        for env_idx, env in enumerate(envs):
            print(f"\nTraining on environment variation {env_idx + 1}/{len(envs)}")
            self.set_env(env)  # Switch to current environment
            remaining_steps = steps_per_env

            while remaining_steps > 0:
                chunk_size = min(CHECKPOINT_INTERVAL, remaining_steps)
                self.learn(
                    total_timesteps=chunk_size,
                    reset_num_timesteps=False
                )

                cumulative_steps += chunk_size
                remaining_steps -= chunk_size

                checkpoint_path = os.path.join(
                    CHECKPOINT_DIR,
                    f"msac_step_{cumulative_steps}.zip"
                )
                self.save(checkpoint_path)
                print(f"Checkpoint saved at {cumulative_steps:,} steps")


# =============================================
# Training Pipeline
# =============================================
def train_and_evaluate(env):
    """Orchestrates training of all three variants"""
    print("\n=== Training ESAC ===")
    esac = ESAC(env)
    esac.train()

    print("\n=== Training MSAC ===")
    msac = MSAC('MlpPolicy', env, verbose=0)
    # Create varied environment instances for meta-training
    randomized_envs = [gym.make("EnergyTradingEnv-v0") for _ in range(3)]
    msac.meta_train(randomized_envs)

    print("\n=== Training SN-SAC ===")
    sn_sac = SAC('MlpPolicy', env,
                 policy_kwargs={'net_arch': [256, 256]},
                 learning_rate=INIT_LR)
    for step in range(0, TIMESTEPS, CHECKPOINT_INTERVAL):
        sn_sac.learn(total_timesteps=CHECKPOINT_INTERVAL, reset_num_timesteps=False)
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"sn_sac_step_{step + CHECKPOINT_INTERVAL}.zip"
        )
        sn_sac.save(checkpoint_path)
        print(f"SN-SAC checkpoint: {checkpoint_path}")

    return sn_sac, esac, msac


# =============================================
# Evaluation Utilities
# =============================================
def evaluate_model(model, env, episodes=32):
    """
    Comprehensive model evaluation
    Returns:
        - Mean reward
        - Reward standard deviation
        - List of individual episode rewards
    """
    print(f"\nEvaluating {model.__class__.__name__}...")
    rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = model.predict(obs)[0]  # Get action from model
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {episode + 1:02d}: {total_reward:+.2f}")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"\nEvaluation Complete:")
    print(f"Mean Reward: {mean_reward:.2f}")
    print(f"Std Deviation: {std_reward:.2f}")

    return mean_reward, std_reward, rewards


# =============================================
# Visualization Utilities
# =============================================
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
    # Environment setup
    print("Initializing Energy Trading Environment...")
    env = gym.make("EnergyTradingEnv-v0")

    # Model training
    #sn_sac, esac, msac = train_and_evaluate(env)

    # Model loading
    print("\nLoading trained models...")
    sn_sac_model = SAC.load(os.path.join("checkpoints", "sn_sac_step_800000.zip"))

    # Load ESAC ensemble
    n_models = 3
    esac_models = []
    for i in range(n_models):
        model_path = os.path.join("checkpoints", f"esac_{i}_step_800000.zip")
        esac_models.append(SAC.load(model_path))


    # ESAC wrapper for unified interface
    class ESACWrapper:
        """Wrapper for ESAC ensemble prediction"""

        def __init__(self, models):
            self.models = models

        def predict(self, observation):
            actions = [model.predict(observation, deterministic=True)[0]
                       for model in self.models]
            return np.mean(actions, axis=0)


    esac_agent = ESACWrapper(esac_models)
    msac_model = MSAC.load(os.path.join("checkpoints", "msac_step_800000.zip"))
    sac_model = SAC.load(os.path.join("checkpoints", "sac_step_800000.zip"))

    # Comparative evaluation
    print("\n=== Starting Comparative Evaluation ===")
    esac_perf, esac_std, esac_rewards = evaluate_model(esac_agent, env)
    msac_perf, msac_std, msac_rewards = evaluate_model(msac_model, env)
    sb_sac_perf, sb_sac_std, sb_sac_rewards = evaluate_model(sn_sac_model, env)
    sac_perf, sac_std, sac_rewards = evaluate_model(sac_model, env)

    # Enhanced visualization
    labels = ["ESAC", "MSAC", "SN-SAC", "SAC"]
    means = [esac_perf, msac_perf, sb_sac_perf, sac_perf]
    stds = [esac_std, msac_std, sb_sac_std, sac_std]
    palette = sns.color_palette("viridis", 3)

    create_comparison_plot(labels, means, stds, palette)
    plt.show()