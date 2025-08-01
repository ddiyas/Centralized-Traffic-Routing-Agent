from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from traffic_env import TrafficEnv
import numpy as np


def make_env(max_cars=400):
    def _init():
        return TrafficEnv(max_cars=max_cars)

    return _init


def evaluate_policy(model, n_episodes=3, max_cars=400):
    print(f"\n--- Evaluating Policy ({n_episodes} episodes, {max_cars} cars) ---")

    rewards = []
    lengths = []
    arrivals = []

    for ep in range(n_episodes):
        env = TrafficEnv(max_cars=max_cars)
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and steps < 100:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        cars_arrived = len([c for c in env.cars.values() if c["status"] == "arrived"])
        final_paths = env.get_final_paths()
        arrival_rate = cars_arrived / max_cars * 100

        rewards.append(total_reward)
        lengths.append(steps)
        arrivals.append(arrival_rate)

        print(
            f"  Episode {ep+1}: Reward={total_reward:.2f}, Steps={steps}, Arrivals={cars_arrived}/{max_cars} ({arrival_rate:.1f}%)"
        )
        print(f"Episode paths: {final_paths}")

    avg_reward = np.mean(rewards)
    avg_length = np.mean(lengths)
    avg_arrival = np.mean(arrivals)

    print(
        f"Average over {n_episodes} episodes: reward = {avg_reward:.2f}, length = {avg_length:.2f}"
    )

    return avg_reward, avg_length, avg_arrival


if __name__ == "__main__":
    MAX_CARS = 400
    TOTAL_TIMESTEPS = 20000
    CHECKPOINT_INTERVAL = 4000
    NUM_ENVS = 4

    print(f"Training Configuration:")
    print(f"  Max Cars: {MAX_CARS}")
    print(f"  Total Timesteps: {TOTAL_TIMESTEPS}")
    print(f"  Parallel Environments: {NUM_ENVS}")

    env = DummyVecEnv([make_env(MAX_CARS) for _ in range(NUM_ENVS)])
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_tensorboard/")

    for step in range(0, TOTAL_TIMESTEPS, CHECKPOINT_INTERVAL):
        print(
            f"\n=== Training from {step} to {step+CHECKPOINT_INTERVAL} timesteps ===\n"
        )
        model.learn(total_timesteps=CHECKPOINT_INTERVAL, reset_num_timesteps=False)
        checkpoint_path = f"ppo_traffic_checkpoint_{step+CHECKPOINT_INTERVAL}.zip"
        model.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        avg_reward, avg_length, avg_arrival = evaluate_policy(
            model, n_episodes=5, max_cars=400
        )

        print(f"Average length: {avg_length:.1f}")
        print(f"Average arrival rate: {avg_arrival:.1f}%")

    env_render = TrafficEnv(max_cars=400)
    print(f"\nFinal Evaluation:")
    final_reward, final_length, final_arrival = evaluate_policy(
        model, n_episodes=3, max_cars=400
    )
