
import argparse

from stable_baselines3 import PPO, A2C, DQN

from env.foggy_env import FoggyDriving
from env.renderer import FoggyDrivingRender
from training.trainer import FoggyDrivingTrainer
from .describe import describe


def main():
    parser = argparse.ArgumentParser(description="FoggyDriving CLI")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["describe", "train", "view"],
        required=True,
        help="Operation mode: describe, train, or view",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="PPO",
        choices=["PPO", "A2C", "DQN"],
        help="RL model type (default: PPO)",
    )

    parser.add_argument(
        "--path",
        type=str,
        default="FoggyDrivingModel",
        help="Path to save/load model",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Training timesteps (only for --mode train)",
    )

    args = parser.parse_args()

    mode = args.mode
    model_type = args.model.upper()
    model_path = args.path

    if mode == "describe":
        #describe()
        return

    if mode == "train":
        print(f"\n--- Training {model_type} ---")
        trainer = FoggyDrivingTrainer(model_type=model_type, model_path=model_path)
        trainer.train(total_timesteps=args.timesteps)
        trainer.evaluate(episodes=50)
        trainer.plot_training_curve()
        trainer.plot_eval_curve()

        env = FoggyDriving()
        renderer = FoggyDrivingRender(env)

        if model_type == "PPO":
            model = PPO.load(trainer.model_path, env=env)
        elif model_type == "A2C":
            model = A2C.load(trainer.model_path, env=env)
        elif model_type == "DQN":
            model = DQN.load(trainer.model_path, env=env)
        else:
            raise ValueError("Invalid model type.")

        renderer.record_gif(model, f"FoggyDriving_{model_type}.gif")
        return

    if mode == "view":
        print(f"\n--- Generating GIF of episode from trained model '{model_path}' ---")

        env = FoggyDriving()
        renderer = FoggyDrivingRender(env)

        if os.path.isfile(model_path):
            model = PPO.load(model_path)
            print("Loaded existing model.")
        else:
            print("No saved model found.")
            return


        if model_type == "PPO":
            model = PPO.load(model_path, env=env)
        elif model_type == "A2C":
            model = A2C.load(model_path, env=env)
        elif model_type == "DQN":
            model = DQN.load(model_path, env=env)
        else:
            raise ValueError("Invalid model type.")

        renderer.record_gif(model, f"FoggyDriving_{model_type}.gif")
        return
