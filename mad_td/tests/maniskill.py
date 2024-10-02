import gymnasium as gym

env = gym.make(
    "PickCube-v1",  # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    num_envs=1,
    obs_mode="state",  # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose",  # there is also "pd_joint_delta_pos", ...
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
