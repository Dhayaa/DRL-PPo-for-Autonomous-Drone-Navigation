import os
import shutil
import numpy as np
from datetime import datetime
import torch

print("🔍 CUDA Available:", torch.cuda.is_available())  
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from drone_env import DroneEnv

# -------------------------------
# File and Logging Configuration
# -------------------------------
policy_path = "ppo_drone_model.zip"
backup_policy_path = f"ppo_drone_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
tensorboard_log_dir = "ppo_tensorboard"
log_rewards_path = "ppo_training_rewards.txt"

if os.path.exists(policy_path):
    try:
        shutil.copy(policy_path, backup_policy_path)
        print("📌 Backup of the previous model saved!")
    except Exception as e:
        print(f"⚠ Error during backup: {e}")

# -------------------------------
# Environment Initialization
# -------------------------------
env = DummyVecEnv([lambda: DroneEnv()])
policy_kwargs = dict(net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])])

# -------------------------------
# Model Loading or New
# -------------------------------
if os.path.exists(policy_path):
    print("📌 Existing model found, loading...")
    model = PPO.load(policy_path, env=env, tensorboard_log=tensorboard_log_dir)
else:
    print("🆕 Creating new PPO model...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir,
                device="cuda", learning_rate=0.0003, gamma=0.90, batch_size=256,
                ent_coef=0.2, clip_range=0.3, vf_coef=0.3, max_grad_norm=0.5,
                n_steps=8192, policy_kwargs=policy_kwargs)

# -------------------------------
# Reward Logging Callback
# -------------------------------
class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_file_path, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.log_file_path = log_file_path

    def _on_step(self) -> bool:
        try:
            rewards = self.locals.get("rewards", None)
            dones = self.locals.get("dones", None)

            if rewards is not None and len(rewards) > 0:
                reward = rewards[0]
                self.episode_rewards.append(reward)
                self.logger.record('step_reward', reward)

            if dones is not None and dones[0]:
                total_reward = sum(self.episode_rewards)
                with open(self.log_file_path, "a") as log_file:
                    log_file.write(f"{self.num_timesteps},{total_reward}\n")
                self.logger.record('episode_reward', total_reward)
                self.logger.dump(self.num_timesteps)
                self.episode_rewards = []
        except Exception as e:
            print(f"⚠ Error while logging rewards: {e}")
        return True

reward_callback = RewardLoggingCallback(log_rewards_path)

# -------------------------------
# Checkpoint Callback
# -------------------------------
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./",
    name_prefix="ppo_drone_checkpoint"
)

# -------------------------------
# Training
# -------------------------------
total_timesteps = 100000
print(f"\n🚀 Starting full training for {total_timesteps} timesteps...")
model.learn(
    total_timesteps=total_timesteps,
    callback=[reward_callback, checkpoint_callback],
    tb_log_name=f"PPO_Full_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    progress_bar=True,
    log_interval=10
)

# -------------------------------
# Save Final Model
# -------------------------------
model.save(policy_path)
print(f"\n✅ Training complete! Final model saved at: {policy_path}")
print(f"🧮 Total Timesteps: {model.num_timesteps}")

# -------------------------------
# Summary
# -------------------------------
stats = {
    "time_elapsed": model.num_timesteps,
    "total_timesteps": total_timesteps,
    "fps": model.logger.name_to_value.get("time/fps", 0),
    "approx_kl": model.logger.name_to_value.get("train/approx_kl", 0),
    "clip_fraction": model.logger.name_to_value.get("train/clip_fraction", 0),
    "clip_range": model.logger.name_to_value.get("train/clip_range", 0),
    "entropy_loss": model.logger.name_to_value.get("train/entropy_loss", 0),
    "explained_variance": model.logger.name_to_value.get("train/explained_variance", 0),
    "learning_rate": model.logger.name_to_value.get("train/learning_rate", 0),
    "loss": model.logger.name_to_value.get("train/loss", 0),
    "n_updates": model.logger.name_to_value.get("train/n_updates", 0),
    "policy_gradient_loss": model.logger.name_to_value.get("train/policy_gradient_loss", 0),
    "std": model.logger.name_to_value.get("train/std", 0),
    "value_loss": model.logger.name_to_value.get("train/value_loss", 0),
}

print("\n📊 Training Summary:")
print("------------------------------------------")
for key, value in stats.items():
    print(f"| {key:22} | {value}")
print("------------------------------------------")

