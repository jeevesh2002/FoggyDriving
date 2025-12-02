


from env.foggy_env import FoggyDriving

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio

from stable_baselines3 import PPO,A2C,DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env


class FoggyDrivingTrainer:
    def __init__( self, model_type="PPO", train_logs= "./train_logs", eval_logs= "./eval_logs",
        best_model= "./best_model", tb_log_dir= "./tb_foggy_grid", model_path= "FoggyDrivingModel",
    ):

        if model_path is None:
            model_path = "FoggyDrivingModel"
        self.model_type = model_type
        self.train_logs = train_logs
        self.eval_logs = eval_logs
        self.best_model = best_model
        self.tb_log_dir = tb_log_dir
        self.model_path = model_path


        os.makedirs(self.train_logs, exist_ok=True)
        os.makedirs(self.eval_logs, exist_ok=True)
        os.makedirs(self.best_model, exist_ok=True)

    def make_env(self,rank: int, log_dir: str = "./train_logs"):
        os.makedirs(log_dir, exist_ok=True)

        def _make():
            env = FoggyDriving()
            return Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.monitor.csv"))

        return _make


    def train(self, total_timesteps = 1_000_000):

        env_fns = [self.make_env(i, log_dir=self.train_logs) for i in range(8)]
        env = DummyVecEnv(env_fns)
        eval_env = Monitor(FoggyDriving(),filename=os.path.join(self.eval_logs, "eval_monitor.monitor.csv"))

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.best_model,
            log_path=self.eval_logs,
            eval_freq=20_000,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
            verbose=1,
        )

        model = None
        if self.model_type == "PPO":

            model = PPO(
                "MlpPolicy",
                env,
                learning_rate = 3e-4,
                n_steps = 2048,
                batch_size = 512,
                clip_range = 0.1,
                gae_lambda = 0.92,
                ent_coef = 0.005,
                n_epochs=10,
                gamma=0.99,
                vf_coef=0.5,
                tensorboard_log=self.tb_log_dir,
                device='cuda',
                verbose=1
            )

        elif self.model_type == "A2C":
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=5,
                gamma=0.99,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log=self.tb_log_dir,
                device='cuda',
                verbose=1
            )

        elif self.model_type == "DQN":
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=1e-3,
                buffer_size=100_000,
                learning_starts=1_000,
                batch_size=64,
                train_freq=4,
                target_update_interval=500,
                gamma=0.99,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                max_grad_norm=10,
                tensorboard_log=self.tb_log_dir,
                device='cuda',
                verbose=1
            )


        print(f"\nTraining {self.model_type} for {total_timesteps:} timesteps\n")
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        print(f"\nModel saved : {self.model_path}\n")
        model.save(self.model_path)
        env.close()

    def evaluate(self, episodes: int = 50):
        env = FoggyDriving()

        if self.model_type == "PPO":
            model = PPO.load(
                self.model_path,
                env=env
            )
        elif self.model_type == "A2C":
            model = A2C.load(
                self.model_path,
                env=env
            )
        elif self.model_type == "DQN":
            model = DQN.load(
            self.model_path,
            env=env
        )

        total_reward = 0.0
        total_len = 0
        collisions = 0

        for _ in range(episodes):
            obs, info = env.reset()
            done = False
            trunc = False
            ep_reward = 0.0
            ep_len = 0

            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(int(action))
                ep_reward += r
                ep_len += 1

            total_reward += ep_reward
            total_len += ep_len
            collisions += int(info.get("collision", False))

        env.close()

        print(f"\nEvaluation : {episodes} episodes:")
        print(f"  Avg return     : {total_reward / episodes}")
        print(f"  Avg ep length  : {total_len / episodes} steps")
        print(f"  Avg Collision  : {collisions / episodes}\n")

    def load_training_logs(self):

        files = sorted(glob.glob(os.path.join(self.train_logs, "monitor_*.monitor.csv")))

        rewards = []
        timesteps = []

        for f in files:
            data = np.genfromtxt(f, skip_header=2, delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)

            rewards.append(data[:, 0])
            timesteps.append(data[:, 2])

        if not rewards:
            return np.array([]), np.array([])

        rewards = np.concatenate(rewards)
        timesteps = np.concatenate(timesteps)

        order = np.argsort(timesteps)
        return timesteps[order], rewards[order]


    def plot_training_curve(self, out_path=None):

        if out_path is None:
            out_path = os.path.join(self.train_logs, f"training_curve_{self.model_type}.png")
        timesteps, rewards = self.load_training_logs()


        plt.figure(figsize=(6, 4))
        plt.plot(timesteps, rewards, label="Expected Return")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode return")
        plt.title("Expected Return vs Timesteps (PPO)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Training curve : {out_path}")

    def plot_eval_curve(self, out_path=None):

        if out_path is None:
            out_path = f"eval_curve_{self.model_type}.png"

        evaluations = os.path.join(self.eval_logs, "evaluations.npz")

        data = np.load(evaluations)
        timesteps = data["timesteps"].flatten()
        results = data["results"]

        mean_reward = results.mean(axis=1)
        std_reward = results.std(axis=1)

        plt.figure(figsize=(6, 4))
        plt.plot(timesteps, mean_reward, label="Evalution mean return")
        plt.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward,
                         alpha=0.2)
        plt.xlabel("Timesteps")
        plt.ylabel("Return")
        plt.title("Mean Return vs Timesteps")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Evaluation curve : {out_path}")
