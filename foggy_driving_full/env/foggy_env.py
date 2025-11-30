
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List
from .renderer import FoggyDrivingRender



class FoggyDriving(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode=None,
        min_speed= 1,
        max_speed= 5,
        max_fog_levels= 2,
        max_range_by_fog=None,
        lidars= 9,
        max_steps= 400,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.renderer = FoggyDrivingRender(self)

        self.grid_width = 2
        self.grid_height = 40
        self.num_lanes = 2

        self.min_speed = min_speed
        self.max_speed = max_speed
        self.ego_speed = float((min_speed + max_speed) / 2)

        self.max_fog_levels = max_fog_levels

        self.fog_levels = [i for i in range(max_fog_levels + 1)]
        if max_range_by_fog is None:
            decay = 0.6
            self.max_range_by_fog = {
                fog: float(self.grid_height) * (decay ** fog)
                for fog in self.fog_levels
            }
        else:
            self.max_range_by_fog = max_range_by_fog

        self.lidars = lidars
        self.beam_angles = np.linspace(-math.pi / 4, math.pi / 4, self.lidars)

        self.max_steps = max_steps

        # Action space: 0 = maintain, 1 = accelerate, 2 = brake, 3 = lane left, 4 = lane right
        self.action_space = spaces.Discrete(5)

        # Observation: 2 lane one-hot + speed + fog + lidar readings
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4 + self.lidars,), dtype=np.float32
        )


        self.rng = np.random.RandomState()


        self.ego_lane = 1
        self.step_count = 0
        self.cars: List[Dict[str, float]] = []

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.ego_lane = int(self.rng.randint(0, self.num_lanes))
        self.ego_speed = float((self.min_speed + self.max_speed) / 2)
        self.fog = int(self.rng.choice(self.fog_levels))


        self.cars = []
        n_cars = int(self.rng.randint(5, 10))

        for i in range(n_cars):
            if i < 3:
                self.cars.append(self._spawn_car(4, 10, self.max_speed - 1))
            else:
                self.cars.append(self._spawn_car(10.0, self.grid_height))

        for i in range(3):
          self.cars.append({"lane": self.ego_lane, "dist": self.rng.uniform(2, 4), "speed": self.min_speed})

        obs = self._get_obs().astype(np.float32)
        return obs, {}

    def _spawn_car(self, dmin: float, dmax: float, smax: int = None) -> Dict[str, float]:
        if smax is None:
            smax = self.max_speed

        lane = int(self.rng.randint(0, self.num_lanes))
        dist = float(self.rng.uniform(dmin, dmax))

        p_fast = 0.1         
        slow_ratio = 0.6     

        if self.rng.rand() < p_fast:
            speed = float(self.rng.uniform(self.ego_speed, smax))
        else:
            target = max(self.min_speed, self.ego_speed * slow_ratio)
            speed = float(self.rng.uniform(self.min_speed, target))

        speed = np.clip(speed, self.min_speed, self.max_speed - 1)

        return {"lane": lane, "dist": dist, "speed": speed}

    def _update_cars(self) -> None:

        for c in self.cars:
            if self.rng.rand() < 0.3:
                c["speed"] = float(
                    np.clip(c["speed"] + self.rng.choice([-1,0,+1]), self.min_speed, self.max_speed - 1)
                )
            if self.rng.rand() < 0.05:
                c["speed"] = max(self.min_speed, c["speed"] - 2)
            if self.rng.rand() < 0.05 and c["dist"]>3:
                c["lane"] = 1-c["lane"]

        new_cars = []
        for c in self.cars:
            rel_speed = self.ego_speed - c["speed"]
            c["dist"] -= rel_speed
            if c["dist"] > -1:
                new_cars.append(c)
        self.cars = new_cars

        if self.rng.rand() < 0.1:
            self.cars.append(
                self._spawn_car(self.grid_height, self.grid_height+5,self.min_speed+1)
            )

        if self.rng.rand() < 0.1:
            self.cars.append(
                self._spawn_car(self.max_range_by_fog[self.fog], self.grid_height, self.max_speed-1)
            )

    def _lidar(self) -> np.ndarray:
        max_r = self.max_range_by_fog[self.fog]
        dists = np.ones(self.lidars, dtype=np.float32) * max_r

        ego_x = self.ego_lane + 0.5
        ego_y = 0.0

        car_boxes = []
        for c in self.cars:
            x0 = c["lane"]
            car_boxes.append((x0, x0 + 1, c["dist"], c["dist"] + 1))

        for i, angle in enumerate(self.beam_angles):
            dx = math.sin(angle)
            dy = math.cos(angle)
            t = 0.0
            step = 0.5
            hit = False
            while t < max_r and not hit:
                t += step
                x = ego_x + dx * t
                y = ego_y + dy * t
                if x < 0 or x >= self.grid_width:
                    break
                if y < 0:
                    continue
                for (xmin, xmax, ymin, ymax) in car_boxes:
                    if xmin <= x < xmax and ymin <= y < ymax:
                        dists[i] = t
                        hit = True
                        break

        noise_scale = 0.02 * (1 + 0.03*self.fog)
        noisy = dists * (1 + self.rng.normal(0, noise_scale, size=dists.shape))
        return np.clip(noisy, 0, max_r).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        lane_onehot = np.zeros(2, dtype=np.float32)
        lane_onehot[self.ego_lane] = 1.0

        speed_norm = np.float32(
            (self.ego_speed - self.min_speed)
            / (self.max_speed - self.min_speed + 1e-8)
        )
        fog_norm = np.float32(self.fog / max(self.fog_levels))

        lidar = self._lidar()
        lidar_norm = lidar / self.max_range_by_fog[self.fog]

        obs = np.concatenate(
            [
                lane_onehot,
                np.array([speed_norm], dtype=np.float32),
                np.array([fog_norm], dtype=np.float32),
                lidar_norm,
            ]
        ).astype(np.float32)
        return obs

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.step_count += 1

        if action == 1:
            self.ego_speed = min(self.max_speed, self.ego_speed + 1)
        elif action == 2:
            self.ego_speed = max(self.min_speed, self.ego_speed - 1)

        if action == 3:
            self.ego_lane = max(0, self.ego_lane - 1)
        elif action == 4:
            self.ego_lane = min(self.num_lanes - 1, self.ego_lane + 1)

        self._update_cars()

        if self.rng.rand() < 0.2:
            self.fog = int(self.rng.choice(self.fog_levels))

        collision = False
        for c in self.cars:
            if c["lane"] != self.ego_lane:
                continue
            if 0 < c["dist"] < 1:
                collision = True
                break

        terminated = collision
        truncated = self.step_count >= self.max_steps

        reward = float(self.ego_speed)
        if collision:
            reward -= 50.0
        if truncated and not terminated:
            reward += 100.0

        obs = self._get_obs().astype(np.float32)
        info = {"collision": collision}
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.renderer.render(self.render_mode)
