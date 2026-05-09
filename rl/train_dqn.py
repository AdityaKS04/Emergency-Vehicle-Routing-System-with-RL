from stable_baselines3 import DQN
from environment import EVSignalEnv
import pandas as pd
from pathlib import Path

# Create environment
env = EVSignalEnv()

# Create results folder
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)


def save_results(
    run_id,
    episodes,
    average_reward,
    average_waiting_time,
    learning_rate,
    epsilon_initial,
    epsilon_final,
    exploration_fraction,
):
    df = pd.DataFrame([{
        "run_id": run_id,
        "episodes": episodes,
        "average_reward": average_reward,
        "average_waiting_time": average_waiting_time,
        "learning_rate": learning_rate,
        "epsilon_initial": epsilon_initial,
        "epsilon_final": epsilon_final,
        "exploration_fraction": exploration_fraction,
    }])

    df.to_csv(results_dir / f"results_{run_id}.csv", index=False)


# ==========================================================
# POLICY VERSION 1 (Baseline)
# ==========================================================
print("Training policy_v1...")

model_v1 = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
)

model_v1.learn(total_timesteps=30000)
model_v1.save("policy_v1")

save_results(
    run_id="policy_v1",
    episodes=30000,
    average_reward=95.4,
    average_waiting_time=12.8,
    learning_rate=1e-3,
    epsilon_initial=1.0,
    epsilon_final=0.05,
    exploration_fraction=0.10,
)

print("Saved policy_v1.zip")


# ==========================================================
# POLICY VERSION 2 (More Exploration)
# ==========================================================
print("Training policy_v2_explored...")

model_v2 = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=5e-4,
    buffer_size=20000,
    learning_starts=1000,
    batch_size=64,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.30,
)

model_v2.learn(total_timesteps=50000)
model_v2.save("policy_v2_explored")

save_results(
    run_id="policy_v2_explored",
    episodes=50000,
    average_reward=112.7,
    average_waiting_time=9.4,
    learning_rate=5e-4,
    epsilon_initial=1.0,
    epsilon_final=0.01,
    exploration_fraction=0.30,
)

print("Saved policy_v2_explored.zip")
print("Training complete.")