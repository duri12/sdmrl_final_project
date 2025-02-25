# # SDMRL Final Project

## Project Overview

This project involves developing and training reinforcement learning (RL) models to optimize energy trading in a simulated market environment. The environment simulates energy demand, price fluctuations, and battery storage management.

## Project Structure

- `electric_market_env.py`: Contains the custom environment `EnergyTradingEnv` for the energy trading simulation.
- `main.py`: Main script for training, saving, loading, and evaluating RL models.
- `README.md`: Project documentation.

## Dependencies

- Python 3.8+
- `gymnasium`
- `numpy`
- `matplotlib`
- `stable-baselines3`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/duri12/sdmrl_final_project.git
    cd sdmrl_final_project
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Training Models

To train the RL models, uncomment the training lines in `main.py` and run the script:
```sh
python main.py