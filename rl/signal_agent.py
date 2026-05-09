from stable_baselines3 import DQN
import numpy as np
from pathlib import Path


# Path to trained DQN model
MODEL_PATH = Path(__file__).resolve().parent / "policy_v2_explored.zip"

# Load trained model once when module is imported
model = DQN.load(MODEL_PATH)


def choose_signal(queue_length, speed, clearance, route, return_debug=False):
    """
    Uses the trained DQN model to choose the best traffic signal.

    If return_debug=True, also returns detailed information
    about the RL state and selected action.
    """

    has_E = 1 if "E" in route else 0
    has_B = 1 if "B" in route else 0

    state = np.array(
        [queue_length, speed, clearance, has_E, has_B],
        dtype=np.float32
    ).reshape(1, -1)

    action, _ = model.predict(state, deterministic=True)
    action = int(np.asarray(action).item())

    chosen_signal = "E" if action == 0 else "B"

    if return_debug:
        debug = {
            "state_vector": state.flatten().tolist(),
            "action": action,
            "action_meaning": chosen_signal,
            "has_E": has_E,
            "has_B": has_B,
        }
        return chosen_signal, debug

    return chosen_signal