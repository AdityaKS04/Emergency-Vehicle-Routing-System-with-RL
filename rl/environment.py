import gymnasium as gym
from gymnasium import spaces
import numpy as np


class EVSignalEnv(gym.Env):
    """
    State:
        [queue_length, speed, clearance_distance, route_contains_E, route_contains_B]

    Actions:
        0 -> choose signal E
        1 -> choose signal B
    """

    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([500, 60, 10, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        queue = np.random.uniform(10, 500)
        speed = np.random.uniform(1, 60)
        clearance = np.random.uniform(0.1, 5.0)

        route_has_E = np.random.randint(0, 2)
        route_has_B = np.random.randint(0, 2)

        if route_has_E == 0 and route_has_B == 0:
            route_has_E = 1

        self.state = np.array(
            [queue, speed, clearance, route_has_E, route_has_B],
            dtype=np.float32
        )

        return self.state, {}

    def step(self, action):
        queue, speed, clearance, has_E, has_B = self.state

        chosen_signal = "E" if action == 0 else "B"

        reward = 0.0

        if chosen_signal == "E" and has_E:
            reward += 100
        elif chosen_signal == "B" and has_B:
            reward += 100
        else:
            reward -= 100

        reward -= clearance * 10
        reward += speed
        reward -= queue * 0.05

        terminated = True
        truncated = False

        return self.state, reward, terminated, truncated, {}