
import math

import matplotlib.pyplot as plt
import numpy as np
import imageio


class FoggyDrivingRender:

    def __init__(self, env):
        self.env = env

    def _draw_figure(self):
        max_r = self.env.max_range_by_fog[self.env.fog]
        H = self.env.grid_height
        W = self.env.grid_width

        fig, ax = plt.subplots(figsize=(3, 6))
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_facecolor("#d3d3d3")

        for x in range(W + 1):
            ax.axvline(x, color="white", linewidth=0.5, alpha=0.6)
        for y in range(H + 1):
            ax.axhline(y, color="white", linewidth=0.5, alpha=0.6)

        car_w = 1
        car_l = 1
        ego_y0 = 1.0
        ego_lane = self.env.ego_lane
        ax.add_patch(
            plt.Rectangle(
                (ego_lane + 0.05, ego_y0),
                car_w - 0.1,
                car_l,
                color="blue",
            )
        )

        for c in self.env.cars:
            x0 = c["lane"]
            y0 = ego_y0 + c["dist"]
            ax.add_patch(
                plt.Rectangle(
                    (x0 + 0.05, y0),
                    car_w - 0.1,
                    car_l,
                    color="red",
                )
            )

        lidar = self.env._lidar()
        ego_cx = ego_lane + 0.5
        ego_cy = ego_y0 + car_l / 2

        for d, ang in zip(lidar, self.env.beam_angles):
            dx = math.sin(ang)
            dy = math.cos(ang)
            x_end = float(np.clip(ego_cx + dx * d, 0, W))
            y_end = float(np.clip(ego_cy + dy * d, 0, max_r))
            ax.plot([ego_cx, x_end], [ego_cy, y_end], color="yellow", linewidth=1.0)

        if self.env.fog > 0:
            fog_start = max_r
            fog_height = H - fog_start
            ax.add_patch(
                plt.Rectangle(
                    (0, fog_start),
                    W,
                    fog_height,
                    color="black",
                    alpha=0.25,
                )
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Fog={self.env.fog}, speed={self.env.ego_speed:.0f}, step={self.env.step_count}"
        )

        fig.tight_layout()
        return fig

    def render(self, mode: str = "rgb_array"):
        fig = self._draw_figure()

        if mode == "human":
            plt.show()
            return None

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = (
            np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            .reshape(h, w, 4)
        )
        plt.close(fig)
        return rgba

    def frame(self):
        return self.render(mode="rgb_array")

    def record_gif(self, model, gif_path: str = "FoggyDriving.gif", max_steps: int = 400):
        frames = []
        obs, info = self.env.reset()
        done = False
        trunc = False
        step = 0

        while not (done or trunc) and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = self.env.step(int(action))
            frames.append(self.frame().copy())
            step += 1

        imageio.mimsave(gif_path, frames, fps=8)
        print(f"GIF saved: {gif_path}")
