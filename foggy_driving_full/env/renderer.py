
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

        # Two-column layout: left = road, right = info panel
        fig = plt.figure(figsize=(5, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])

        ax = fig.add_subplot(gs[0, 0])
        info_ax = fig.add_subplot(gs[0, 1])

        # ---- Left: road view ----
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_facecolor("#d3d3d3")

        # Road grid lines
        for x in range(W + 1):
            ax.axvline(x, color="white", linewidth=0.5, alpha=0.6)
        for y in range(H + 1):
            ax.axhline(y, color="white", linewidth=0.5, alpha=0.6)

        car_w = 1.0
        car_l = 1.0
        ego_y0 = 1.0
        ego_lane = self.env.ego_lane

        # Ego car (blue)
        ax.add_patch(
            plt.Rectangle(
                (ego_lane + 0.05, ego_y0),
                car_w - 0.1,
                car_l,
                color="blue",
            )
        )

        # Other cars (red)
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

        # Lidar beams (yellow)
        lidar = self.env._lidar()
        ego_cx = ego_lane + 0.5
        ego_cy = ego_y0 + car_l / 2.0

        for d, ang in zip(lidar, self.env.beam_angles):
            dx = math.sin(ang)
            dy = math.cos(ang)
            x_end = float(np.clip(ego_cx + dx * d, 0, W))
            y_end = float(np.clip(ego_cy + dy * d, 0, max_r))
            ax.plot([ego_cx, x_end], [ego_cy, y_end], color="yellow", linewidth=1.0)

        # Fog visualization above max range
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
            f"Fog={self.env.fog}, step={self.env.step_count}",
        )

        # ---- Right: info panel (HUD) ----
        info_ax.axis("off")
        info_ax.set_title("Ego state", loc="left")

        ego_speed = self.env.ego_speed
        speed_norm = (
            ego_speed - self.env.min_speed
        ) / (self.env.max_speed - self.env.min_speed + 1e-8)

        y = 0.95
        info_ax.text(0.05, y, f"Speed: {ego_speed:.1f}", fontsize=9)
        y -= 0.08
        info_ax.text(0.05, y, f"Speed norm: {speed_norm:.2f}", fontsize=8)
        y -= 0.08
        info_ax.text(0.05, y, f"Lane: {self.env.ego_lane}", fontsize=9)
        y -= 0.08
        info_ax.text(0.05, y, f"Fog level: {self.env.fog}", fontsize=9)
        y -= 0.08
        info_ax.text(0.05, y, f"Step: {self.env.step_count}", fontsize=9)

        # Nearest few cars info
        info_ax.text(0.05, y - 0.08, "Nearest cars:", fontsize=9, fontweight="bold")
        y -= 0.16

        sorted_cars = sorted(
            [c for c in self.env.cars if c["dist"] >= 0.0],
            key=lambda c: c["dist"],
        )

        for i, c in enumerate(sorted_cars[:4]):
            rel_speed = c["speed"] - ego_speed
            info_ax.text(
                0.05,
                y,
                f"{i+1}) lane {c['lane']}, d={c['dist']:.1f}, Δv={rel_speed:+.1f}",
                fontsize=8,
            )
            y -= 0.08

        # Legend at the bottom
        y -= 0.05
        info_ax.text(0.05, y, "Blue = ego", fontsize=8, color="blue")
        y -= 0.06
        info_ax.text(0.05, y, "Red = traffic", fontsize=8, color="red")
        y -= 0.06
        info_ax.text(0.05, y, "Yellow = lidar", fontsize=8, color="goldenrod")

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
