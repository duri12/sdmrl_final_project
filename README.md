# SDMRL Final Project

## Project Overview 📊

This project involves developing and training reinforcement learning (RL) models to optimize energy trading in a simulated market environment. The environment simulates energy demand, price fluctuations, and battery storage management.

**Key Features**:
- Custom Gymnasium environment simulating electricity market dynamics
- Implementation of SAC (Soft Actor-Critic) and its variants
- Comparative analysis of RL algorithms (PPO, SAC,TD3)
- Advanced generalization techniques:
  - Spectral Normalization SAC (SN-SAC)
  - Ensemble SAC (ESAC)
  - Meta-Learning SAC (MSAC)
  - 
## Project Structure🗂️

- `electric_market_env.py`: Contains the custom environment `EnergyTradingEnv` for the energy trading simulation.
- `main.py`: Main script for training, saving, loading, and evaluating RL models.
- `part_3.py`: Implements and compares three SAC variants - SN-SAC, ESAC, and MSAC.
- `plot_learning_prog.py`: Script to plot the learning progression of RL models.
- `utils/fix_file_names.py`: Utility script to fix file names in the checkpoints directory.
- `README.md`: Project documentation.

## Dependencies🧩

- Python 3.8+
- `gymnasium`
- `numpy`
- `matplotlib`
- `stable-baselines3`
- `seaborn`
- `torch`
- `torchvision`

## Installation⚙️

1. Clone the repository:
    ```sh
    git clone https://github.com/duri12/sdmrl_final_project.git
    cd sdmrl_final_project
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage 🚀

### Training Models

To train the RL models, uncomment the training lines in `main.py` and run the script:
```sh
python main.py
```
### Evaluating Models
To evaluate the trained models, run the script:
```sh
python main.py 
```

### Plotting Learning Progression
To plot the learning progression of RL models, run the script:
```sh
python plot_learning_prog.py
```
